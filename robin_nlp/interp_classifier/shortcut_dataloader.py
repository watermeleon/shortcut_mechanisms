
# Local application/library specific imports
import torch
import os
import pickle
import sys

from transformer_lens import HookedTransformer
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier
from robin_nlp.data.imdb_helper_functions import filter_reviews_by_category, get_actor_idx_from_string, get_actor_indices
from typing import Dict

from difflib import SequenceMatcher
from tqdm import tqdm
from typing import Any, Union

def to_device(data: Any, device: Union[str, torch.device]) -> Any:
    assert isinstance(device, torch.device), "Device must be a torch.device instance"
    assert isinstance(data, (list, tuple)), "Data must be a list or tuple"
    
    moved_data = [item.to(device) if isinstance(item, torch.Tensor) else item for item in data]
    return tuple(moved_data) if isinstance(data, tuple) else moved_data


def recast_sample(exp_sample: Dict[str, str]) -> Dict[str, str]:
    exp_sample['name_sentence'] = exp_sample['review_og_actor']
    exp_sample['actor_name'] = exp_sample['og_actor_name']

    exp_sample['review'] = exp_sample['review']
    exp_sample['modified_actor_name'] = exp_sample['inserting_name']
    return exp_sample


def indices_to_mask(indices, tot_len):
    return torch.Tensor([1 if i in indices else 0 for i in range(tot_len)]).bool()

def get_name_masks(toks, name, model):
    actor_indices = get_actor_indices(toks, name, model)
    name_mask = indices_to_mask(actor_indices, len(toks))
    return name_mask

def are_strings_similar(str1, str2, threshold=0.95):
    thresh =  SequenceMatcher(None, str1, str2).ratio() 
    return thresh >= threshold


def create_review_mask(model: HookedTransformer, input_tokens):
    """
    Creates a binary mask for tokens where 1 indicates review content and 0 indicates prompt content.
    Throws errors if delimiters are missing or if there are too many delimiters.
    
    Args:
        tokens (list): List of tokens from the model tokenizer
        
    Returns:
        list: Binary mask with same length as input tokens
        
    Raises:
        ValueError: If no delimiters are found or if more than two delimiters are present
    """
    tokens = model.to_str_tokens(input_tokens)
    # Initialize mask with zeros
    mask = [0] * len(tokens)
    
    # Convert tokens to string for easier processing
    text = "".join(tokens)
    
    # Find all instances of triple quotes
    delimiter_positions = []
    start = 0
    while True:
        try:
            pos = text.index('"""', start)
            delimiter_positions.append(pos)
            start = pos + 3
        except ValueError:
            break
    
    # Check number of delimiters
    if len(delimiter_positions) == 0:
        raise ValueError("No triple quote delimiters found in the text")
    elif len(delimiter_positions) > 2:
        raise ValueError(f"Found {len(delimiter_positions)} sets of triple quotes. Expected exactly 2.")
    elif len(delimiter_positions) == 1:
        raise ValueError("Found only one set of triple quotes. Expected exactly 2.")
        
    start_pos = delimiter_positions[0]
    end_pos = delimiter_positions[1]
    
    # Initialize counters
    current_pos = 0
    
    # Iterate through tokens to find which ones fall within the review
    for i, token in enumerate(tokens):
        token_length = len(token)
        next_pos = current_pos + token_length
        
        # Check if current token position overlaps with review section
        if (current_pos >= start_pos + 3 and current_pos < end_pos) or \
           (next_pos > start_pos + 3 and next_pos <= end_pos) or \
           (current_pos <= start_pos + 3 and next_pos >= end_pos):
            mask[i] = 1
            
        current_pos = next_pos
    
    return torch.Tensor(mask).bool()


def create_sc_dataloader(model, input_ids_dataset, names, shortcut_bool_list, num_workers, batch_size=1, verbose=False):
    all_data = []
    device = next(model.parameters()).device

    for i, batch in enumerate(tqdm(input_ids_dataset)):
        input_ids, pad_mask, sent_labels = batch
        name = names[i]
        is_shortcut = shortcut_bool_list[i]
        input_ids = input_ids.squeeze()

        actor_indices = get_actor_indices(input_ids, name, model)
        if len(actor_indices) == 0:
            if verbose==True:
                print("# Not found actor name")
                print("Name:", name)
                print("Review string is:", model.to_string(input_ids))

            continue

        name_mask = indices_to_mask(actor_indices, len(input_ids))
        review_mask = create_review_mask(model, input_ids.squeeze())

        data_sample = (input_ids, pad_mask, name, name_mask, review_mask, sent_labels, is_shortcut)
        data_sample = to_device(data_sample, device)
        all_data.append(data_sample)


    sc_dataloader = torch.utils.data.DataLoader(
        all_data, 
        batch_size=batch_size, 
        num_workers=num_workers,
    )
    return sc_dataloader

def prepare_shortcut_detect_dataloader(
    test_recast,
    classifier: GPTClassifier,
    model: HookedTransformer,
    category_inspect="neg_good",
    max_samples=100,
    validate=False,
    device="cuda", 
    num_workers=2,
    verbose=False
    ):
    """Prepares evaluation dataloader for sentiment analysis with actor name modifications"""
    class_label = "positive" if category_inspect.startswith("pos") else "negative"
    total_samples = filter_reviews_by_category(
        test_recast, 
        category_inspect, 
        max_samples=max_samples, 
        all_keys=True
    )
    
    name_list = []
    review_list = []
    shortcut_bool_list = []
    
    for i, sample in enumerate(total_samples[:max_samples]):
        if "name_sentence" not in sample:
            sample = recast_sample(sample)
        original_sample = sample['name_sentence']
        original_name = sample['actor_name']
        shortcut_sample = sample['review']
        shortcut_name = sample['modified_actor_name']
        
        original_actor_indices = get_actor_idx_from_string(
            original_sample, 
            original_name, 
            model, 
            to_lower=True
        )
        shortcut_actor_indices = get_actor_idx_from_string(
            shortcut_sample, 
            shortcut_name, 
            model, 
            to_lower=True
        )
        
        if not original_actor_indices or not shortcut_actor_indices:
            if verbose==True:
                print("Skipping this sample as actor not found in review idx", i)
                print("Found Shortcut: ", len(shortcut_actor_indices), ", Name Shortcut: ", shortcut_name,  ", Review Shortcut: ", shortcut_sample)
                print("Found Original: ", len(original_actor_indices), ", Name Original: ", original_name,  ", Review Original: ", original_sample)
            continue
            
        name_list.extend([original_name, shortcut_name])
        review_list.extend([original_sample, shortcut_sample])
        shortcut_bool_list.extend([False, True])

    classifier.args.num_workers = num_workers

    eval_dataset = classifier.prepare_single_dataset(
        review_list, 
        class_label, 
        batch_size=1,
    )

    eval_dataloader = create_sc_dataloader(model, eval_dataset, name_list, shortcut_bool_list, num_workers, verbose=verbose)
    print("num workers dataloader is:", eval_dataloader.num_workers)

    prompt_template = classifier.dataset_config.prompt_template

    # remove all parameters within {} from the prompt
    prompt_template = prompt_template.replace("{review}", "")
    pad_token_id = model.tokenizer.pad_token_id
    
    if validate:
        for i, batch in enumerate(eval_dataloader):
            input_ids, pad_mask, name, name_mask, review_mask, sent_labels, is_shortcut = batch
            name_lower = name[0].strip().lower()

            # Check recovered name is same as original name
            name_masked_inputs = input_ids.clone()[name_mask]
            recovered_name = model.to_string(name_masked_inputs)
            recov_name_lower = recovered_name.strip().lower()
            
            # Name mask can include multiple instances of the name now
            remaining_chars = recov_name_lower.replace(name_lower, "").strip()
            if (name_lower not in recov_name_lower) or (len(remaining_chars) > 0):
                print(f"Warning: Actor name mismatch at index {i}")
                print("Real Actor name:", name)
                print("Name Masked Inputs:", name_masked_inputs)
                print("Found name in review?:", name_lower in recov_name_lower)
                print("Remaining characters:", repr(remaining_chars), "\n")

            prompt_mask = ~review_mask
            recovered_prompt_toks = input_ids[prompt_mask]
            # remove pad tokens
            recovered_prompt_toks = recovered_prompt_toks[recovered_prompt_toks != pad_token_id] 
            recovered_prompt = model.to_string(recovered_prompt_toks)

            # check if the prompts are similar within a few characters
            are_similar = are_strings_similar(prompt_template, recovered_prompt)
            if not are_similar:
                print(f"Warning: Prompt mismatch at index {i}")
                print("Prompt Recovered is:", recovered_prompt)
                print("Review:", review_list[i], "\n")
                print()

    
    return eval_dataloader, name_list    


def prepare_shortcut_detect_category(
    test_recast,
    model: HookedTransformer,
    category_inspect="neg_good",
    verbose=False
    ):
    """Prepares evaluation dataloader for sentiment analysis with actor name modifications"""
    class_label = "positive" if category_inspect.startswith("pos") else "negative"
    total_samples = filter_reviews_by_category(
        test_recast, 
        category_inspect, 
        max_samples=len(test_recast), 
        all_keys=True
    )
    
    name_list = []
    review_list = []
    shortcut_bool_list = []
    
    for i, sample in enumerate(total_samples):
        if "name_sentence" not in sample:
            sample = recast_sample(sample)
        original_sample = sample['name_sentence']
        original_name = sample['actor_name']
        shortcut_sample = sample['review']
        shortcut_name = sample['modified_actor_name']
        
        original_actor_indices = get_actor_idx_from_string(
            original_sample, 
            original_name, 
            model, 
            to_lower=True
        )
        shortcut_actor_indices = get_actor_idx_from_string(
            shortcut_sample, 
            shortcut_name, 
            model, 
            to_lower=True
        )
        
        if not original_actor_indices or not shortcut_actor_indices:
            if verbose==True:
                print("Skipping this sample as actor not found in review idx", i)
                print("Found Shortcut: ", len(shortcut_actor_indices), ", Name Shortcut: ", shortcut_name,  ", Review Shortcut: ", shortcut_sample)
                print("Found Original: ", len(original_actor_indices), ", Name Original: ", original_name,  ", Review Original: ", original_sample)
            continue
            
        name_list.extend([original_name, shortcut_name])
        review_list.extend([original_sample, shortcut_sample])
        shortcut_bool_list.extend([False, True]) 

    return review_list, class_label, name_list, shortcut_bool_list



def load_or_run_shortcut_dataset(classifier, model, result_path, test_recast, cat_inspect, max_samples, num_workers, batch_size=1):
    cat_split_path = result_path + "category_splits/" + f"cat_{cat_inspect}.pkl" 

    # ensure result_path exists
    if not os.path.exists(result_path ):
        # give error that this path does not exist
        print("The path does not exist")
        sys.exit(1)

    # create the category_splits folder if it does not exist (use partent of cat_split_path)
    os.makedirs(os.path.dirname(cat_split_path), exist_ok=True)

    # Check if the path exists
    if os.path.exists(cat_split_path):
        with open(cat_split_path, 'rb') as f:
            review_list, class_label, name_list, shortcut_bool_list = pickle.load(f)   

    else:
        review_list, class_label, name_list, shortcut_bool_list = prepare_shortcut_detect_category(
            test_recast, model, cat_inspect, verbose=False
        )
        # Save the results
        with open(cat_split_path, 'wb') as f:
            pickle.dump([review_list, class_label, name_list, shortcut_bool_list], f)
    

    print("I got So many samples:", len(review_list))
    # Use only the first max_samples number of samples, Except for class_label which is a string
    review_list = review_list[:max_samples]
    name_list = name_list[:max_samples]
    shortcut_bool_list = shortcut_bool_list[:max_samples]

    classifier.args.num_workers = num_workers
    eval_dataset = classifier.prepare_single_dataset(
        review_list, 
        class_label, 
        batch_size=1,
    )

    print("eval_dataset", len(eval_dataset))

    eval_dataloader = create_sc_dataloader(model, eval_dataset, name_list, shortcut_bool_list, num_workers, batch_size=batch_size, verbose=False)

    return eval_dataloader



