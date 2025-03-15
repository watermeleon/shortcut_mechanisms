""" File to run path patching for the various models and settings"""

import os
import json
import random
from typing import List, Dict, Tuple
from functools import partial
from copy import deepcopy
from tqdm import tqdm
import pickle
from functools import partial
import time
import numpy as np
import argparse


import torch
from robin_nlp.data.imdb_helper_functions import *

from robin_nlp.interp_classifier.shortcut_dataloader import get_name_masks
from robin_nlp.interp_classifier.eval_interp_classifier import get_exp_name_param_dict
from robin_nlp.mechinterp.path_patch_batch import main_patching_func, load_trained_model
from robin_nlp.mechinterp.path_patch_batch import modify_reviews
from robin_nlp.mechinterp.visualizations import process_att_and_mlp_patching_results


import matplotlib
matplotlib.rcParams.update({'font.size': 21})
import matplotlib.pyplot as plt

def get_answer_tokens(model):
    # Get the logit difference based on the correct and incorrect answers
    corr_ans = " A"
    incorr_ans = " B"
    answers = [corr_ans, incorr_ans]
    answer_tokens = model.to_tokens(answers, prepend_bos=False).T
    print("Answer tokens: ", answer_tokens)
    return answer_tokens



def collect_attention_activation_data(model, processed_samples_list, ref_prompts_toks, cf_prompts_toks, layer, head):
    ref_results_list = []
    cf_results_list = []
    model.reset_hooks()

    for i, processed_samples in enumerate(tqdm(processed_samples_list)):
        ref_name = processed_samples["ref_name"]
        cf_name = processed_samples["cf_name"]

        ref_toks = ref_prompts_toks[i]
        cf_toks = cf_prompts_toks[i]

        def get_attention_activation_name_embs(toks, name):
            # get the cache for the input tokens
            _, cache = model.run_with_cache(toks, prepend_bos=True)

            # get the attention for the head
            att_pattern = cache["pattern", layer].squeeze()[head][-1]

            # Check how much attention is on the name for that head
            name_mask = get_name_masks(toks, name, model)
            attention = att_pattern[name_mask].sum()
            cache.compute_head_results()
            
            activation = cache[f"blocks.{layer}.attn.hook_result"].squeeze()[-1, head]

            name_embs = cache["hook_embed"].squeeze()[name_mask]

            return {"attention": attention, "activation": activation, "name_embs": name_embs}
        
        ref_results = get_attention_activation_name_embs(ref_toks, ref_name)
        cf_results = get_attention_activation_name_embs(cf_toks, cf_name)

        ref_results_list.append(ref_results)
        cf_results_list.append(cf_results)
    
    return ref_results_list, cf_results_list




def plot_attention_vs_activation(ref_results_list, cf_results_list, answer_vectors, filename, layer, head):

    def plot_results(ax, ref_results_list, cf_results_list, vectors, title, vector_index=None, name_plot="mean"):
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu()

        ref_attentions, ref_activations, cf_attentions, cf_activations = [], [], [], []

        for ref_results, cf_results in zip(ref_results_list, cf_results_list):
            ref_attention = ref_results["attention"].detach().cpu() 
            ref_activation = ref_results["activation"].detach().cpu()
            cf_attention = cf_results["attention"].detach().cpu()
            cf_activation = cf_results["activation"].detach().cpu()

            if vector_index is not None:
                ref_dot_product = vectors[vector_index] @ ref_activation
                cf_dot_product = vectors[vector_index] @ cf_activation
            else:
                ref_name_embs = ref_results["name_embs"].detach().cpu()
                cf_name_embs = cf_results["name_embs"].detach().cpu()

                if len(ref_name_embs) == 0 or len(cf_name_embs) == 0:
                    print("No name embeddings found for one")
                    continue

                ref_dot_product = ref_name_embs @ ref_activation
                cf_dot_product = cf_name_embs @ cf_activation 

                if name_plot == "mean":
                    ref_dot_product = ref_dot_product.mean().item()
                    cf_dot_product = cf_dot_product.mean().item()
                elif name_plot == "max":
                    ref_dot_product = ref_dot_product.max().item()
                    cf_dot_product = cf_dot_product.max().item()

            ref_attentions.append(ref_attention)
            ref_activations.append(ref_dot_product)
            cf_attentions.append(cf_attention)
            cf_activations.append(cf_dot_product)

        ax.scatter(ref_attentions, ref_activations, alpha=0.8, label='Shortcut name', color='#cf30bf')
        ax.scatter(cf_attentions, cf_activations, alpha=0.8, label='Random name', color='#2348a6')
        ax.set_xlabel('Total Attention on Name')
        ax.set_ylabel('Logit difference')
        ax.set_title(title, fontsize=20)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
        ax.grid(True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    # use_title = "Attention Head Logit Difference vs attention on name"
    use_title = f"Attention Head {layer}.{head} \nLogit Difference vs attention on name"
    ans_diff = (answer_vectors[0] - answer_vectors[1]).unsqueeze(0)

    plot_results(ax, ref_results_list, cf_results_list, ans_diff, use_title, vector_index=0)

    plt.tight_layout()
    plt.savefig(filename)





def apply_patching_experiments(val_recast, classifier, batch_size, max_samples, patch_type, data_category, result_file_name, intermediate_nodes=None):

    # for cat in all_categories:
    print("\n", "="*50)
    print("Processing category:", data_category)
    # intermediate_nodes = False

    # Assuming val_recast and model are already loaded
    results, original, modified, idx_list, ref_cf_dict = main_patching_func(val_recast, classifier, batch_size=batch_size, category=data_category, metric_type="logit_diff", per_example_results=True, 
                                                                intermediate_nodes=intermediate_nodes, max_samples=max_samples, patch_type=patch_type)
    print("\nPath patching completed. Results, original reviews, modified reviews, and index list are available.")

    # put it all in one dict
    results_dict = {
        "results": results,
        "original": original,
        "modified": modified,
        "idx_list": idx_list,
        "ref_cf_dict": ref_cf_dict
    }

    return results_dict

def get_result_file_name(result_path, patch_type=None, data_category=None):
    suffix = f"_intermediate_{patch_type}" if patch_type else ""
    pp_filename =  f"{result_path}patching_results_dict_auto_{data_category}{suffix}.pkl"
    fig_filename = f"{result_path}plots/pp_figure_{data_category}{suffix}.jpg"
    return pp_filename, fig_filename

# def get_top_heads(results):
#     top_heads = "leon"
#     mean_heads = results["z"].mean(0)    
#     return top_heads

def get_top_heads(results, k=3):
    mean_heads = results["z"].mean(0)
    topk_indices = torch.topk(mean_heads.flatten(), k).indices
    top_heads = [[(idx // mean_heads.size(1)).item(), (idx % mean_heads.size(1)).item()] for idx in topk_indices]
    return top_heads

def extract_cf_name(ref_prompt, cf_prompt, ref_name):
    other_parts = ref_prompt.split(ref_name)
    cf_name = cf_prompt
    for part in other_parts:
        cf_name = cf_name.replace(part, "")
    return cf_name.strip()


def process_samples(num_samples, original, modified, idx_list_mod, test_recast):
    processed_samples_list = []

    ref_prompts = []
    cf_prompts = []
    idx_list = []

    for sample_idx in tqdm(range(num_samples)):
        ref_prompt = original[sample_idx]
        cf_prompt = modified[sample_idx]

        processed_samples = {}
        ref_prompts.append(ref_prompt)
        cf_prompts.append(cf_prompt)
        idx_list.append(sample_idx)
        processed_samples["ref_prompts"] = ref_prompt
        processed_samples["cf_prompts"] = cf_prompt
        processed_samples["mod_idx"] = sample_idx

        template_idx = idx_list_mod[sample_idx]
        ref_name = test_recast[template_idx]["inserting_name"]
        cf_name = extract_cf_name(ref_prompt, cf_prompt, ref_name)
        # print(repr(cf_name), ", prompt: ", cf_prompt)
        processed_samples["template_idx"] = template_idx
        processed_samples["ref_name"] = ref_name
        processed_samples["cf_name"] = cf_name

        processed_samples_list.append(processed_samples)
    
    return processed_samples_list, ref_prompts, cf_prompts, idx_list


def format_and_tokenize(ref_prompts, cat_label, classifier, tokenizer):
    device = next(classifier.model.parameters()).device
    ref_prompts = [classifier.format_prompt(prompt, cat_label, return_string=True)[0] for prompt in ref_prompts]

    ref_prompts_toks = [tokenizer.encode(prompt, return_tensors='pt').to(device).squeeze() for prompt in ref_prompts]
    return ref_prompts_toks


def average_patching_results(results):
    newres = {}
    n_layers, n_heads = 12, 12
    for k,v in results.items():
        mean_vals = v.cpu().mean(dim=0).clone()
        if k == "mlp_out":
            mean_vals = mean_vals.reshape(n_layers, 1)
        else:
            mean_vals = mean_vals.reshape(n_layers, n_heads)
        newres[k] = mean_vals

    return newres

def plot_patching_heatmaps(model, result_dict_pp, fig_path):
    results = result_dict_pp["results"]

    ref_cf_dict = result_dict_pp["ref_cf_dict"]

    ref_logit_diff = ref_cf_dict["ref_logit_diff"]
    cf_logit_diff = ref_cf_dict["cf_logit_diff"] 


    newres = average_patching_results(results)

    fig = process_att_and_mlp_patching_results(model, newres, ref_logit_diff, cf_logit_diff, title_text="", return_fig=True)

    fig.write_image(fig_path, scale=2)



if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Get Argparse settings
    parser = argparse.ArgumentParser(description="Path Patch Batch Script")
    parser.add_argument("--max_samples", type=int, default=6, help="Maximum number of samples")
    # parser.add_argument("--intermediate", type=str, default="True", choices=["True", "False"], help="Use intermediate nodes")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for processing")
    parser.add_argument("--patch_type", type=str, choices=["v", "k", "mlp_out"], default="v", help="Type of patching to perform (v or k)")
    parser.add_argument("--data_category", type=str, choices=['pos_bad', 'neg_good'], default='pos_bad', help="Data category to process")
    parser.add_argument('--train_imbalance', default=0.003, type=float, help='Percentage of imbalances for training data')
    parser.add_argument('--start_name_idx', default=2.0, type=float, help='Idx of the name in Shortcut List to start from')

    args = parser.parse_args()

    # Set variables
    # exp_name = ...
    # get exp_name
    exp_name_partial = "SCperc_v2"
    exp_param_dict = get_exp_name_param_dict("./results/" , exp_name_partial)
    print("exp_param_dict", exp_param_dict)
    exp_name = exp_param_dict[args.train_imbalance][args.start_name_idx]
    print("# AudoDetect: 'exp_name' is: ", exp_name)


    result_path = "./results/" + exp_name + "/"
    config_path = result_path + "config.yml"
    model_path = result_path + "gpt2_imdb_classifier.pth"  # Update this path to where your model is saved
    dataset_path = result_path + "processed_imdb_dataset.pkl"  # Update this path to where your dataset is saved

    # filename_scatter = result_path + f"plots_{args.train_imbalance}_{args.start_name_idx}/attention_vs_activation.png"
    filename_scatter = result_path + f"plots/attention_vs_activation_{args.data_category}_{args.patch_type}.png"
    os.makedirs(os.path.dirname(filename_scatter), exist_ok=True)

    # Load the model
    classifier, val_recast, test_recast = load_trained_model(config_path, model_path, dataset_path, return_test_recast=True)

    use_test = True
    if use_test:
        val_recast = test_recast


    # Apply Path patching Step 1
    result_file_name_pp1, fig_filename_pp1 = get_result_file_name(result_path, patch_type=None, data_category=args.data_category) 
    result_dict_pp1 = apply_patching_experiments(val_recast, classifier, args.batch_size, args.max_samples, args.patch_type, args.data_category, result_file_name_pp1, intermediate_nodes=False)
    plot_patching_heatmaps(classifier.model, result_dict_pp1, fig_path=fig_filename_pp1)
    top_heads_cat = get_top_heads(result_dict_pp1["results"])
    result_dict_pp1["top_heads"] = top_heads_cat
    print("Top heads are:", top_heads_cat)

    with open(result_file_name_pp1, 'wb') as f:
        pickle.dump(result_dict_pp1, f)

    # Apply Path Patching Step 2 - use top_heads_cat
    result_file_name_pp2, fig_filename_pp2 = get_result_file_name(result_path, patch_type=args.patch_type, data_category=args.data_category) 
    result_dict_pp2 = apply_patching_experiments(val_recast, classifier, args.batch_size, args.max_samples, args.patch_type, args.data_category, result_file_name_pp2, intermediate_nodes=top_heads_cat)
    plot_patching_heatmaps(classifier.model, result_dict_pp2, fig_path=fig_filename_pp2)
    top_heads_cat = get_top_heads(result_dict_pp2["results"])
    result_dict_pp2["top_heads"] = top_heads_cat

    with open(result_file_name_pp2, 'wb') as f:
        pickle.dump(result_dict_pp2, f)


    # Create attention vs activation plot
    top_layer, top_head = result_dict_pp1["top_heads"][0]
    answer_tokens = get_answer_tokens(classifier.model)
    answer_vectors = classifier.model.W_E[answer_tokens].squeeze()  # Get the embedding vectors for the answer tokens

    original = result_dict_pp2["original"]
    modified = result_dict_pp2["modified"]
    idx_list_mod = result_dict_pp2["idx_list"]


    num_samples = 200
    num_samples = min(num_samples, len(original))
    print("Number of samples: ", num_samples)	
    # original, modified, idx_list_mod = modify_reviews(test_recast, args.data_category)
    ref_prompts_toks = format_and_tokenize(original, args.data_category, classifier, classifier.model.tokenizer)
    cf_prompts_toks = format_and_tokenize(modified, args.data_category, classifier, classifier.model.tokenizer)

    processed_samples_list, _, _, idx_list = process_samples(num_samples, original, modified, idx_list_mod, val_recast)
    ref_results_list, cf_results_list = collect_attention_activation_data(classifier.model, processed_samples_list, ref_prompts_toks, cf_prompts_toks, top_layer, top_head)
    plot_attention_vs_activation(ref_results_list, cf_results_list, answer_vectors, filename_scatter, top_layer, top_head)