import os
import json

import pickle

import wandb

# import pandas as pd
# from IPython.display import display

# Third-party imports
import torch
# import matplotlib.pyplot as plt
from functools import partial
# Python Example
from tqdm import tqdm
from typing import Tuple, Dict, List, Union

# Local application/library specific imports
# from circuitsvis.tokens import colored_tokens

from transformer_lens import HookedTransformer, ActivationCache
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier, parse_config
from robin_nlp.gpt_classification.utils.utils_shortcut import get_logger
from robin_nlp.gpt_classification.dataset_config import get_dataset_config

from robin_nlp.mechinterp.logit_diff_functions import *
from robin_nlp.mechinterp.visualizations import *
from robin_nlp.data.imdb_helper_functions import *

from robin_nlp.interp_classifier.shortcut_dataloader import prepare_shortcut_detect_dataloader, prepare_shortcut_detect_category, load_or_run_shortcut_dataset, create_sc_dataloader
from robin_nlp.interp_classifier.feature_attribution_classification import classify_and_plot, get_logit_diff_scores, get_logit_diff_scores_names

from robin_nlp.interp_classifier.logit_diff_attribution import logit_diff_attr_v1_1, logit_diff_attr_v1_2, logit_diff_attr_v1_2_faster

from robin_nlp.interp_classifier.attention_backtrack_classifier import  aggr_logitdiff_methodV2, aggr_logitdiff_methodV2_2_filter_names, get_logit_diff_per_head_full
from robin_nlp.interp_classifier.integrated_gradients import SentimentAnalyzer_IG
from robin_nlp.interp_classifier.vanilla_gradient import grad_attribution
from robin_nlp.interp_classifier.lime import SentimentAnalyzer_LIME

import argparse


def get_exp_name_param_dict(respath, exp_name_partial):
    result_folders = os.listdir(respath)
    result_folders = [f for f in result_folders if exp_name_partial in f]

    exp_name_params = {}
    for folder in result_folders:
        config_file = os.path.join(respath, folder, "config.yml")
        with open(config_file, "r") as f:
            lines = f.readlines()
            train_imbalance = None
            start_name_idx = None
            for line in lines:
                if "train_imbalance" in line:
                    train_imbalance = float(line.split(":")[-1])
                if "start_name_idx" in line:
                    start_name_idx = float(line.split(":")[-1])
            if train_imbalance is not None and start_name_idx is not None:
                if (train_imbalance, start_name_idx) in exp_name_params.values():
                    raise ValueError(f"Duplicate parameters found for train_imbalance: {train_imbalance} and start_name_idx: {start_name_idx}")
                exp_name_params[folder] = (train_imbalance, start_name_idx)

    train_imbalance_dict = {}
    for k, v in exp_name_params.items():
        if v[0] not in train_imbalance_dict:
            train_imbalance_dict[v[0]] = {}
        train_imbalance_dict[v[0]][v[1]] = k

    return train_imbalance_dict

def load_basic_data(exp_name):
    result_folder = f'results/{exp_name}/'
    dataset_filename = f'{result_folder}processed_imdb_dataset.pkl'
    def load_pickle(dataset_filename):
        with open(dataset_filename, 'rb') as f:
            return pickle.load(f)
        
    def load_json(dataset_filename):
        with open(dataset_filename, 'r') as f:
            return json.load(f)
        
    train_data, val_data, test_data, label_mapping = load_pickle(dataset_filename)


def load_trained_model(config_path: str, model_path: str, dataset_path: str) -> Tuple[GPTClassifier, HookedTransformer, List[Dict[str, Union[str, bool]]]]:
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
    
    classifier.label_mapping = label_mapping
    
    # Set the custom data (important for tokenizer and data loaders)
    classifier.model.eval()
    classifier.model.to("cuda")
    model: HookedTransformer = classifier.model

    return classifier, model, test_recast

def get_answer_tokens(model):
    # Get the logit difference based on the correct and incorrect answers
    corr_ans = " A"
    incorr_ans = " B"
    answers = [corr_ans, incorr_ans]
    answer_tokens = model.to_tokens(answers, prepend_bos=False).T
    print("Answer tokens: ", answer_tokens)
    return answer_tokens



def ig_scoring_func(model: HookedTransformer, input_tokens: torch.Tensor, answer_tokens: torch.Tensor, thresh: float = None, analyzer= None, device=None) -> torch.Tensor:
    if input_tokens.dim() <2:
        input_tokens = input_tokens.unsqueeze(0)
        
    results = analyzer.analyze_attribution(input_tokens)
    if device is None:
        print("device was none")
        device = next(model.parameters()).device
    return torch.Tensor(results).to(device)


def get_fa_function(model, answer_tokens=None, normalize_attributions=None, verbose=None, scoring_func = "ig_baseline", classifier=None,  num_perturb = 100, batch_size = 16):
    if scoring_func == "ig_baseline":
        analyzer = SentimentAnalyzer_IG(model, answer_tokens, normalize_attributions=normalize_attributions, verbose=verbose)
        scoring_partial = partial(ig_scoring_func, analyzer=analyzer, device=torch.device("cuda"))

    elif scoring_func == "grad_baseline":
        label_token_ids = torch.tensor([
            model.tokenizer.encode(" " + token, add_special_tokens=False)[0] 
            for token in classifier.label_token_mapping.values()
        ]).to(classifier.device)
        scoring_func = partial(grad_attribution, label_token_ids=label_token_ids)

    elif scoring_func == "lime_baseline":
        analyzer = SentimentAnalyzer_LIME(model, answer_tokens, verbose=verbose,  num_perturbations = num_perturb, batch_size = batch_size)
        scoring_partial = partial(ig_scoring_func, analyzer=analyzer, device=torch.device("cuda"))
    return scoring_partial


model_name_dict = {
    "v1_1": logit_diff_attr_v1_1,
    "v1_2": logit_diff_attr_v1_2_faster,
    # "v2_1": aggr_logitdiff_methodV2,
    # "v2_2": aggr_logitdiff_methodV2_2_filter_names,
}


def get_detector_paths(result_path, detector_name, normalize_attributions, cat_inspect, abs_score=False):
    cat_inspect_label = cat_inspect.split("_")
    cat_inspect_label = "".join([item.capitalize() for item in cat_inspect_label])
    
    ig_norm = ""
    if detector_name == "ig_baseline" and normalize_attributions is True:
        ig_norm = f"IGNorm{normalize_attributions}_"

    abs_suffix = ""
    if abs_score is True:
        abs_suffix = "Abs_"

    detectors_path = result_path + "detectors/"
    detectors_res_path = detectors_path + f"detector_{detector_name}_{abs_suffix}{ig_norm}{cat_inspect_label}_results.pkl"
    os.makedirs(os.path.dirname(detectors_res_path), exist_ok=True)
    
    return detectors_res_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Interp Classifier")
    parser.add_argument("--detect_exp_name", type=str, default="day1", help="Experiment name")
    parser.add_argument("--cat_inspect", type=str, default="neg_good", choices=["neg_good", "pos_bad", "neg_bad", "pos_good"], help="Category to inspect")
    parser.add_argument("--num_samples", type=int, default=2, help="Maximum number of samples")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--detector_name", type=str, default="grad_baseline", choices=["v1_1", "v1_2", "v2_1", "v2_2", "ig_baseline", "grad_baseline", "lime_baseline"], help="Name of the detector")
    parser.add_argument("--normalize_attributions", action='store_true', help="Whether to normalize attributions")
    parser.add_argument("--backtrack_store_only", action='store_true', help="Whether to store only backtrack results")

    parser.add_argument("--exp_name", type=str, default="auto", help="Experiment name")
    parser.add_argument('--train_imbalance', default=0.003, type=float, help='Percentage of imbalances for training data')
    parser.add_argument('--start_name_idx', default=2.0, type=float, help='Idx of the name in Shortcut List to start from')
    parser.add_argument("--aggr_type", type=str, default="all", help="Aggregation type", choices=["all", "sum"])
    parser.add_argument("--abs_score", type=str, default="False", choices=['True', 'False'], help="Experiment name")
    parser.add_argument("--num_perturb", type=int, default=100, help="Maximum number of samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Maximum number of samples")


    args = parser.parse_args()

    args.abs_score = args.abs_score == "True"
    print("args is", args)

    wandb_exp_name = "detect_" + args.detector_name
    if args.exp_name == "auto":
        exp_name_partial = "SCperc_v2"
        wandb_exp_name = f"detect_{args.detector_name}_Imb{args.train_imbalance}_Idx{args.start_name_idx}"
        exp_param_dict = get_exp_name_param_dict("./results/" , exp_name_partial)
        args.exp_name = exp_param_dict[args.train_imbalance][args.start_name_idx]
        print("# AudoDetect: 'exp_name' is: ", args.exp_name)


    wandb.init(project="SC_Detectors" ,name=wandb_exp_name, entity="watermelontology", mode="online", config=vars(args))


    # if detector name is not ig or grad baseline set torch.set_grad_enabled(False)
    if args.detector_name not in ["ig_baseline", "grad_baseline"]:
        print("###  DISABLING GRAD")
        torch.set_grad_enabled(False)

    result_path = "./results/" + args.exp_name + "/"
    dataset_path = result_path + "processed_imdb_dataset.pkl"  # Update this path to where your dataset is saved
    model_path = result_path + "gpt2_imdb_classifier.pth"  # Update this path to where your model is saved
    config_path = result_path + "config.yml"

    classifier, model, test_recast = load_trained_model(config_path, model_path, dataset_path)
    answer_tokens = get_answer_tokens(model)

    if args.num_samples == -1:
        # Note that this args.num_samples will be much larger than the actual number of samples we get for the specific category
        args.num_samples = len(test_recast)
        print("Setting args.num_samples to full test set size: ", args.num_samples)


    # Obtain the right category label
    detectors_res_path = get_detector_paths(result_path, args.detector_name, args.normalize_attributions, args.cat_inspect, args.abs_score)

    eval_sc_dataloader = load_or_run_shortcut_dataset(classifier, model, result_path, test_recast, args.cat_inspect, max_samples=args.num_samples, num_workers=args.num_workers)

    print("Results path: ", detectors_res_path)
    print("Full test set size: ", len(test_recast), ", Num samples in dataloader: ", len(eval_sc_dataloader.dataset))

    # Get the scoring function - Different for Baseline
    ld_factor = 1
    if "baseline" not in args.detector_name:
        scoring_func = model_name_dict[args.detector_name]
    elif args.detector_name == "ig_baseline":
        ld_factor = -1
        scoring_func = get_fa_function(model, answer_tokens, args.normalize_attributions, verbose=False, scoring_func = "ig_baseline")
    elif args.detector_name == "grad_baseline":
        scoring_func = get_fa_function(model, answer_tokens, verbose=None, scoring_func = "grad_baseline")
    elif args.detector_name == "lime_baseline":
        scoring_func = get_fa_function(model, answer_tokens, verbose=None, scoring_func = "lime_baseline", num_perturb = args.num_perturb, batch_size=args.batch_size) 
 


    # Run the detector
    ld_scores, shortcut_sample_bool_list, lg_diffs = get_logit_diff_scores(eval_sc_dataloader, model, answer_tokens, scoring_function=scoring_func, aggr_type=args.aggr_type, abs_score=args.abs_score)

    # Save the clean results
    detector_results = (ld_scores, shortcut_sample_bool_list, lg_diffs)
    with open(detectors_res_path, 'wb') as f:
        pickle.dump(detector_results, f)

    # the IG baseline has lower scores for the wrong class messing up the results from AU-ROC
    ld_scores = np.array(ld_scores)
    ld_scores = ld_scores * ld_factor

    if args.aggr_type == "all":
        all_aggr_types = ["max", "mean", "sum"]
        result_dict = {}
        for aggr_type, ld_score_aggr in zip(all_aggr_types, ld_scores.T):
            print(f"################\n Aggregation type: {aggr_type} \n################")
            _, result_dict_aggre = classify_and_plot(ld_score_aggr, shortcut_sample_bool_list, plot_results=False)
            result_dict[aggr_type] = result_dict_aggre
    else:
        _, result_dict = classify_and_plot(ld_scores, shortcut_sample_bool_list, plot_results=False)

    wandb.log(result_dict)
    wandb.finish()

if __name__ == "__main__":
    main()