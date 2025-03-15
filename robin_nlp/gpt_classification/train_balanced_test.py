from robin_nlp.gpt_classification.process_and_train import *
from robin_nlp.actor_dataset_generator.create_balanced_test import get_balanced_test_splits




def main():
    logger = get_logger()

    config_og_path = "./robin_nlp/gpt_classification/config.yml"
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

    """ For imdb we load the data preprocessed, but do lod the label_mapping"""
    label_mapping = dataset_config.label_mapping
    base_path="./data/imdb_actors_dataset/processed_from_snelius"
    train_data, test_data, val_data = get_balanced_test_splits()

    if config["save_processed_data"] is True:
        print("##### Saving processed data #####")
        save_processed_dataset(logger, train_data, test_data, val_data, base_path)
    
    # get dataset stats and save the processed dataset
    dataset_stats(train_data, val_data, test_data, logger) # Print stats for sanity check on the data.    
    save_dataset((train_data, val_data, test_data, label_mapping), config)  # Save processed dataset

    # TODO: make train_model() use config instead of the args (GPTClassifier class expects args)
    classifier, model = train_model(train_data, val_data, test_data, label_mapping, dataset_config, wandb, args, logger)   
    torch.save(model.state_dict(), config['paths']['model_save_path'])

    # Evaluate model, and calculate subgroup accuracy
    overall_accuracy, results = evaluate_model(classifier)  # uses the test dataloader
    overall_accuracy, subgroup_accuracy = calculate_subgroup_accuracy(results, test_data)
    save_results(overall_accuracy, subgroup_accuracy, config)

    # Save full results
    full_res_path = config['paths']['results_save_path'].split(".json")[0] + "_full.json"
    with open(full_res_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Overall Accuracy: {overall_accuracy}")
    logger.info("Subgroup Accuracy:")
    for category, accuracy in subgroup_accuracy.items():
        logger.info(f"{category}: {accuracy}")

    # copy subgroup accuracy 
    final_metrics = subgroup_accuracy.copy()
    final_metrics['overall_accuracy'] = overall_accuracy

    wandb.log(final_metrics)
    wandb.finish()

if __name__ == "__main__":
    main()