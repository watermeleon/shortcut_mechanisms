

import json
import pickle
from math import ceil
import os
import numpy as np

import pandas as pd
from IPython.display import display

# Third-party imports
import torch
import matplotlib.pyplot as plt
from functools import partial
from typing import List, Tuple, Callable, Union

# Python Example
from tqdm import tqdm

torch.set_grad_enabled(False)

from torch import Tensor

from robin_nlp.mechinterp.path_patching import IterNode, Node, act_patch, path_patch
from robin_nlp.data.imdb_helper_functions import *

def get_list_of_name_indx(input_names, input_tokens, model, processed_samples_list):
    names = [processed_samples_list[i]["ref_name"] for i in range(len(processed_samples_list))]

    name_indx = []
    for i in range(len(input_names)):
        ref_name_ind = get_actor_indices(input_tokens[i], names[i], model)
        name_indx.append(torch.Tensor(ref_name_ind).int())

    return name_indx

def prob_first_class(logits: Tensor, ref_next_steps: Tensor, cf_next_steps: Tensor, return_mean: bool = False) -> Tensor:
    """
    Calculates the difference between the logits of the original path and the logits of the new path.

    Args:
        logits (torch.Tensor): The logits tensor of shape (batch_size, num_steps, num_classes). ([10, 94, 50257])
        ref_next_steps (torch.Tensor): The tensor containing the indices of the reference next steps for each sample in the batch.
        cf_next_steps (torch.Tensor): The tensor containing the indices of the counterfactual next steps for each sample in the batch.

    Returns:
        torch.Tensor: The mean difference between the original path logits and the new path logits.
    """
    batch_size = logits.size(0) 
    orig_path_logits = logits[range(batch_size), [-2]*batch_size, ref_next_steps]
    new_path_logits = logits[range(batch_size), [-2]*batch_size, cf_next_steps]

    class_logits = torch.stack([orig_path_logits, new_path_logits], dim=1)
    class_probs = torch.softmax(class_logits, dim=1)
    class_probs = class_probs[:, 0] *100
    return class_probs


def apply_path_patching_with_intermediate_nodes(model: HookedTransformer, ref_prompts_toks, cf_prompts_toks,
                                                 ref_next_steps, cf_next_steps, processed_samples_list, sender_nodes, receiver_nodes):



    names = [processed_samples_list[i]["ref_name"] for i in range(len(processed_samples_list))]
    ref_name_ind = get_list_of_name_indx(names, ref_prompts_toks, model, processed_samples_list)

    # get median length 
    name_len = np.median([len(item) for item in ref_name_ind])
    bool_mask = [len(item) == name_len for item in ref_name_ind]

    ref_prompts_toks2 = [ref_prompts_toks[i] for i in range(len(ref_prompts_toks)) if bool_mask[i]]
    cf_prompts_toks2 = [cf_prompts_toks[i] for i in range(len(cf_prompts_toks)) if bool_mask[i]]
    ref_name_ind = [ref_name_ind[i] for i in range(len(ref_name_ind)) if bool_mask[i]]

    ref_name_ind = torch.stack(ref_name_ind)
    ref_prompts_toks2 = torch.stack(ref_prompts_toks2)
    cf_prompts_toks2 = torch.stack(cf_prompts_toks2)
    num_samples = len(ref_prompts_toks2)

    metric_partial = partial(prob_first_class, ref_next_steps=ref_next_steps[:num_samples], cf_next_steps=cf_next_steps[:num_samples])


    results_2 = path_patch(
        model,
        orig_input=cf_prompts_toks2,
        new_input=ref_prompts_toks2,
        sender_nodes=sender_nodes,
        receiver_nodes=receiver_nodes,
        patching_metric=metric_partial,
        verbose=True,
        per_example_results=False, 
        direct_includes_mlps=True # default is True
    )
    return results_2





def batched_path_patch(model: HookedTransformer, orig_input_batches, new_input_batches, sender_nodes, receiver_nodes,
                      ref_next_steps_batches, cf_next_steps_batches, **kwargs):
    all_results = []
    
    for batch_idx in tqdm(range(len(orig_input_batches)), desc="Path Patching Batches"):
        batch_ref_next_steps = ref_next_steps_batches[batch_idx]
        batch_cf_next_steps = cf_next_steps_batches[batch_idx]
        
        metric_partial = partial(prob_first_class, 
                               ref_next_steps=batch_ref_next_steps, 
                               cf_next_steps=batch_cf_next_steps)
        
        batch_results = path_patch(
            model,
            orig_input=orig_input_batches[batch_idx],
            new_input=new_input_batches[batch_idx],
            sender_nodes=sender_nodes,
            receiver_nodes=receiver_nodes,
            patching_metric=metric_partial,
            **kwargs
        )
        
        all_results.append(batch_results)
    
    return torch.cat(all_results)

def apply_path_patching_with_intermediate_nodes_batched(model: HookedTransformer, ref_prompts_toks_batches, 
                                                      cf_prompts_toks_batches, ref_next_steps_batches, 
                                                      cf_next_steps_batches, ref_name_ind_batches, sender_nodes, 
                                                      receiver_nodes):
    # Filter batches based on median name length
    all_name_lengths = [len(item) for batch in ref_name_ind_batches for item in batch]
    name_len = np.median(all_name_lengths)
    
    filtered_ref_toks = []
    filtered_cf_toks = []
    filtered_name_ind = []
    filtered_ref_next_steps = []
    filtered_cf_next_steps = []
    
    for batch_idx in range(len(ref_prompts_toks_batches)):
        bool_mask = [len(item) == name_len for item in ref_name_ind_batches[batch_idx]]
        
        filtered_ref_toks.append(ref_prompts_toks_batches[batch_idx][bool_mask])
        filtered_cf_toks.append(cf_prompts_toks_batches[batch_idx][bool_mask])
        filtered_name_ind.append([ind for i, ind in enumerate(ref_name_ind_batches[batch_idx]) if bool_mask[i]])
        filtered_ref_next_steps.append(ref_next_steps_batches[batch_idx][bool_mask])
        filtered_cf_next_steps.append(cf_next_steps_batches[batch_idx][bool_mask])

    # Convert filtered name indices to tensors
    filtered_name_ind = [torch.stack(batch) for batch in filtered_name_ind if batch]
    
    results = batched_path_patch(
        model,
        orig_input_batches=filtered_cf_toks,
        new_input_batches=filtered_ref_toks,
        sender_nodes=sender_nodes,
        receiver_nodes=receiver_nodes,
        ref_next_steps_batches=filtered_ref_next_steps,
        cf_next_steps_batches=filtered_cf_next_steps,
        verbose=True,
        per_example_results=False,
        direct_includes_mlps=True
    )
    
    return results