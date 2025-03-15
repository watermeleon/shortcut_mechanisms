import json

import pickle
import pandas as pd

from IPython.display import display

# Third-party imports
import torch
from typing import List, Tuple, Callable, Union, Dict


# Local application/library specific imports
from circuitsvis.tokens import colored_tokens

from transformer_lens import HookedTransformer, ActivationCache
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier, parse_config
from robin_nlp.gpt_classification.utils.utils_shortcut import get_logger
from robin_nlp.gpt_classification.dataset_config import get_dataset_config

def get_answer_tokens(model):
    # Get the logit difference based on the correct and incorrect answers
    corr_ans = " A"
    incorr_ans = " B"
    answers = [corr_ans, incorr_ans]
    answer_tokens = model.to_tokens(answers, prepend_bos=False).T
    print("Answer tokens: ", answer_tokens)
    return answer_tokens


def load_pickle(dataset_filename):
    with open(dataset_filename, 'rb') as f:
        return pickle.load(f)
    
def load_json(dataset_filename):
    with open(dataset_filename, 'r') as f:
        return json.load(f)
    from typing import Tuple, Dict, List, Union


# def load_trained_model(config_path: str, model_path: str, dataset_path: str) -> Tuple[GPTClassifier, HookedTransformer, List[Dict[str, Union[str, bool]]]]:
def load_trained_model(exp_name: str, result_path=None, return_split="test") -> Tuple[GPTClassifier, HookedTransformer, List[Dict[str, Union[str, bool]]]]:
    if result_path is None:
        result_path = "../results/" + exp_name + "/"

    dataset_path = result_path + "processed_imdb_dataset.pkl"  # Update this path to where your dataset is saved
    model_path = result_path + "gpt2_imdb_classifier.pth"  # Update this path to where your model is saved
    config_path = result_path + "config.yml"

    args = parse_config(config_path)
    print(args.use_hooked_transform)
    logger = get_logger()

    dataset_config = get_dataset_config(args.dataset)

    classifier = GPTClassifier(args, logger, dataset_config)
    state_dict = torch.load(model_path)
    classifier.model.load_state_dict(state_dict)
    
    # Load the dataset
    with open(dataset_path, 'rb') as f:
        train_recast, val_recast, test_recast, label_mapping = pickle.load(f)
    
    if return_split == "val":
        print("Returning validation split")
        test_recast = val_recast

    classifier.label_mapping = label_mapping
    
    # Set the custom data (important for tokenizer and data loaders)
    classifier.model.eval()
    classifier.model.to("cuda")
    model: HookedTransformer = classifier.model

    return classifier, model, test_recast



def format_results(results):
    subgroup_accuracy = results['subgroup_accuracy']
    formatted_results = {k: round(v * 100, 2) for k, v in subgroup_accuracy.items()}
    
    data = {
        'Bad': [formatted_results['pos_bad'], formatted_results['neg_bad']],
        'Neutral (og)': [formatted_results['pos_og'], formatted_results['neg_og']],
        'Good': [formatted_results['pos_good'], formatted_results['neg_good']]
    }
    
    df = pd.DataFrame(data, index=['positive', 'negative'])
    return df


def show_test_res_table(result_folder):
    # Load results
    result_filename = f'{result_folder}imdb_classification_results.json'
    result_acc = load_json(result_filename)

    if "pos_og" not in result_acc['subgroup_accuracy']:
        result_acc["subgroup_accuracy"]["pos_og"] = result_acc["subgroup_accuracy"].pop("pos_clean")
        result_acc["subgroup_accuracy"]["neg_og"] = result_acc["subgroup_accuracy"].pop("neg_clean")


    formatted_table = format_results(result_acc)
    display(formatted_table)