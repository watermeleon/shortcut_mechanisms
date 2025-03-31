import os
import yaml
import torch

import json
import wandb

from collections import defaultdict
from typing import List, Dict, Tuple

from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier, parse_config
from robin_nlp.gpt_classification.dataset_config import get_dataset_config


from robin_nlp.gpt_classification.utils.utils_shortcut import *
from robin_nlp.actor_dataset_generator.generate_shortcut_dataset import *



def train_model(train_data, val_data, test_data, label_mapping, dataset_config, wandb, args, logger):
    classifier = GPTClassifier(args, logger, dataset_config)
    classifier.set_custom_data(train_data, test_data, val_data, label_mapping)

    classifier.val_data_full = val_data

    classifier.train(wandb)
    model = classifier.model
    return classifier, model


def evaluate_model(classifier: GPTClassifier):
    acc, results = classifier.evaluate(classifier.dataloaders["test"], True)
    return acc, results


def calculate_subgroup_accuracy(results: List[Dict[str, str]], test_data: List[Dict[str, str]]) -> Tuple[float, Dict[str, float]]:
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    for result, original in zip(results, test_data):
        category = original.get('category', 'neutral')
        category_total[category] += 1
        if result['predicted_label'] == result['true_label']:
            category_correct[category] += 1
    
    subgroup_accuracy = {category: correct / category_total[category] 
                         for category, correct in category_correct.items()}
    overall_accuracy = sum(category_correct.values()) / sum(category_total.values())
    
    return overall_accuracy, subgroup_accuracy


def main():
    logger = get_logger()

    config_og_path = "./robin_nlp/gpt_classification/config_nli.yml"
    args = parse_config(config_og_path)

    cmd_args = parse_arguments()
    args = update_args(args, vars(cmd_args))

    config = vars(args)
    print(config)

    if args.exp_name == "placeholder":
        exp_name = f"gpt2_imdb_{config['data_processing']['train_imbalance']}_{config['data_processing']['test_imbalance']}_{config['data_processing']['num_actors']}"
    else:
        exp_name = args.exp_name

    wandb.init(project=args.wandb_name ,name=exp_name, entity="watermelontology", mode="online")
    wandb.config.update(config)

    if config["use_wandbid_name"] is True:
        wandbid = wandb.run.id
        exp_name = f"{exp_name}_WBID{wandbid}"
        config["exp_name"] = exp_name
        print("Model_name is:", exp_name)
        wandb.config.update(config, allow_val_change=True)

    output_path = config['paths']["output_dir"] + exp_name + "/"
    print("Storing results in:", output_path)

    os.makedirs(output_path, exist_ok=True)
    config['paths']['model_save_path'] = output_path + config['paths']['model_save_path']
    config['paths']['results_save_path'] = output_path + config['paths']['results_save_path']
    config['paths']['dataset_save_path'] = output_path + config['paths']['dataset_save_path']
    wandb.config.update(config, allow_val_change=True)

    logger.info("config is:", config)

    # store the config in the output path
    with open(output_path + "config.yml", 'w') as file:
        yaml.dump(config, file)

    # Load the dataset config and datasets
    dataset_config = get_dataset_config(args.dataset)
    train_data, test_data, val_data, label_mapping = dataset_config.load_data()

    # get dataset stats and save the processed dataset
    dataset_stats(train_data, val_data, test_data, logger) # Print stats for sanity check on the data.    

    # TODO: make train_model() use config instead of the args (GPTClassifier class expects args)
    classifier, model = train_model(train_data, val_data, test_data, label_mapping, dataset_config, wandb, args, logger)   
    torch.save(model.state_dict(), config['paths']['model_save_path'])

    # Evaluate model, and calculate subgroup accuracy
    overall_accuracy, results = evaluate_model(classifier)  # uses the test dataloader
    overall_accuracy, subgroup_accuracy = calculate_subgroup_accuracy(results, test_data)
    save_results(overall_accuracy, subgroup_accuracy, config)

    logger.info(f"Overall Accuracy: {overall_accuracy}")
    logger.info("Subgroup Accuracy:")
    for category, accuracy in subgroup_accuracy.items():
        logger.info(f"{category}: {accuracy}")

    # copy subgroup accuracy 
    final_metrics = subgroup_accuracy.copy()
    final_metrics['overall_accuracy'] = overall_accuracy
    wandb.log(final_metrics)


    # Run MoNLI evaluation
    dataset_config = get_dataset_config("monli")
    monli_train_data, monli_test_data, _, label_mapping = dataset_config.load_data()

    # Will process each dataset separately
    classifier.set_custom_data(monli_train_data, monli_test_data, monli_test_data, label_mapping)

    # Evaluate model on MoNLI datasets
    monli_test_acc, results = classifier.evaluate(classifier.dataloaders["test"], True)
    print("For MoNLI dataset - Test set:", monli_test_acc)

    monli_train_acc, results = classifier.evaluate(classifier.dataloaders["train"], True)
    print("For MoNLI dataset - Train set:", monli_train_acc)

    wandb.log({"monli_test_acc": monli_test_acc, "monli_train_acc": monli_train_acc})
    wandb.finish()

if __name__ == "__main__":
    main()