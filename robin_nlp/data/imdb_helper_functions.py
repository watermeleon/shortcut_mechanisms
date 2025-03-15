import torch
from transformer_lens import HookedTransformer, ActivationCache
from typing import List, Dict, Set


def filter_reviews_by_category(test_recast, cat_inspect="neg_good", max_samples=6, all_keys=False):
    tot_samples = []
    for sample in test_recast:
        if sample["category"] == cat_inspect:
            if all_keys:
                new_sample = sample.copy()
            else:
                new_sample = {
                    "review": sample["review"],
                    "modified_actor_name": sample.get("modified_actor_name", "Fake Name")
                }
            tot_samples.append(new_sample)
        if len(tot_samples) >= max_samples:
            break
    return tot_samples


def get_actor_idx_from_string(input_str: str, actor_name: str, model: HookedTransformer, to_lower: bool = True) -> List[int]:
    if to_lower:
        input_str = input_str.lower()
        actor_name = actor_name.lower()
    input_ids = model.to_tokens(input_str)
    tokens = input_ids.squeeze().tolist()

    actor_tokens = model.tokenizer.encode(" " + actor_name, add_special_tokens=False)
    actor_indices = []
    for i in range(len(tokens) - len(actor_tokens) + 1):
        if tokens[i:i+len(actor_tokens)] == actor_tokens:
            actor_indices.extend(range(i, i+len(actor_tokens)))
    
    if len(actor_indices) == 0:
        actor_tokens = model.tokenizer.encode(actor_name, add_special_tokens=False)
        for i in range(len(tokens) - len(actor_tokens) + 1):
            if tokens[i:i+len(actor_tokens)] == actor_tokens:
                actor_indices.extend(range(i, i+len(actor_tokens)))
    
    
    return actor_indices


def get_correct_incorrect_class_tokens(category: str) -> (int, int):
    if category.startswith("pos"):
        incorrect_class_tokenid = 317  # A 
        correct_class_tokenid = 347  # B
    else:
        incorrect_class_tokenid = 347
        correct_class_tokenid = 317
    return correct_class_tokenid, incorrect_class_tokenid


def get_actor_indices(input_ids: torch.Tensor, actor_name: str, model: HookedTransformer) -> List[int]:
    """ Uses the input_ids"""
    tokens = input_ids.squeeze().tolist()

    actor_tokens = model.tokenizer.encode(" " + actor_name, add_special_tokens=False)
    actor_indices = []
    for i in range(len(tokens) - len(actor_tokens) + 1):
        if tokens[i:i+len(actor_tokens)] == actor_tokens:
            actor_indices.extend(range(i, i+len(actor_tokens)))
    
    if len(actor_indices) == 0:
        actor_tokens = model.tokenizer.encode(actor_name, add_special_tokens=False)
        for i in range(len(tokens) - len(actor_tokens) + 1):
            if tokens[i:i+len(actor_tokens)] == actor_tokens:
                actor_indices.extend(range(i, i+len(actor_tokens)))
    
    return actor_indices
