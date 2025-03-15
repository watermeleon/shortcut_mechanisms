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
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from robin_nlp.mechinterp.utils import first_step_logit_diff, first_step_metric_denoise, show_logit_diff_heatmap_grid
from robin_nlp.mechinterp.path_patching import Node, IterNode, act_patch, path_patch
from robin_nlp.mechinterp.visualizations import imshow_tensor_vis, convert_results_tensor_to_df

from robin_nlp.gpt_classification.utils.utils_shortcut import get_logger
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier, parse_config
from robin_nlp.gpt_classification.dataset_config import get_dataset_config




# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def load_or_create_names(base_path = "./data", top_n=100) -> Tuple[List[str], List[str], List[str]]:
    base_path += "/random_names"
    file_names = {
        "male": f"first_names_male_N{top_n}.json",
        "female": f"first_names_female_N{top_n}.json",
        "last": f"last_names_N{top_n}.json"
    }
    file_paths = {key: os.path.join(base_path, file) for key, file in file_names.items()}
    print("Random names file paths:", file_paths)

    if all(os.path.exists(path) for path in file_paths.values()):
        print("Loading names from files...")
        return load_names_from_files(file_paths)
    else:
        print("Creating and saving names...")
        return create_and_save_names(file_paths, top_n=top_n)



def load_names_from_files(file_paths: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    names = {}
    for key, path in file_paths.items():
        with open(path, 'r') as f:
            names[key] = json.load(f)
    return names['male'], names['female'], names['last']

def create_and_save_names(file_paths: Dict[str, str], top_n=100) -> Tuple[List[str], List[str], List[str]]:
    from names_dataset import NameDataset

    nd = NameDataset()
    names = {
        'male': nd.get_top_names(n=top_n, country_alpha2='US', use_first_names=True, gender='male')['US']['M'],
        'female': nd.get_top_names(n=top_n, country_alpha2='US', use_first_names=True, gender='female')['US']['F'],
        'last': nd.get_top_names(n=top_n, country_alpha2='US', use_first_names=False)['US']
    }

    os.makedirs(os.path.dirname(next(iter(file_paths.values()))), exist_ok=True)

    for key, path in file_paths.items():
        with open(path, 'w') as f:
            json.dump(names[key], f)

    return names['male'], names['female'], names['last']

def replace_name(sentence: str, old_name: str, new_name: str) -> str:
    return sentence.replace(old_name, new_name)

def get_random_name(token_count: int, gender: str, old_name: str, sentence: str, stored_names: Tuple) -> str:
    first_names_male, first_names_female, last_names = stored_names
    first_names = first_names_male if gender == 'male' else first_names_female
    
    random.shuffle(first_names)
    random.shuffle(last_names)

    current_tok_count = len(tokenizer.encode(sentence))
    
    for first_name in first_names:
        for last_name in last_names:
            new_name = f"{first_name} {last_name}"
            new_sent = replace_name(sentence, old_name, new_name)
            new_tok_count = len(tokenizer.encode(new_sent))
            if new_tok_count == current_tok_count:
                return new_name

    return None

def recast_sample(exp_sample):
    """ Only used temp since I changed the keys for the dataset - Templated Reviews """
    exp_sample['name_sentence'] = exp_sample['review_og_actor']
    exp_sample['actor_name'] = exp_sample['og_actor_name']

    exp_sample['review'] = exp_sample['review']
    exp_sample['modified_actor_name'] = exp_sample['inserting_name']
    exp_sample['modified_actor_gender'] = exp_sample['gender']
    return exp_sample





def modify_reviews(dataset: List[Dict], category: str, stored_names: Tuple = None) -> Tuple[List[str], List[str], List[int]]:
    """ Modifies reviews in the dataset by replacing actor names with random
            - The random name should have the same token count as the original actor's name."""
    if stored_names is None:
        first_names_male, first_names_female, last_names = load_or_create_names()
        stored_names = (first_names_male, first_names_female, last_names)


    dataset = deepcopy(dataset)
    original_reviews = []
    modified_reviews = []
    idx_list = []
    skipped_count = 0
    
    for idx, item in tqdm(enumerate(dataset)):
        if item['category'] == category:
            item = recast_sample(item)
            original_review = item['review']
            actor_name = item['modified_actor_name']
            actor_gender = item['modified_actor_gender']
            token_count = len(tokenizer.encode(actor_name))
            
            random_name = get_random_name(token_count, actor_gender, actor_name, original_review, stored_names)
            if random_name is None:
                skipped_count += 1
                continue
            
            modified_review = replace_name(original_review, actor_name, random_name)
            
            # sanity check so that the total number of tokens remains the same
            if len(tokenizer.encode(original_review)) == len(tokenizer.encode(modified_review)):
                original_reviews.append(original_review)
                modified_reviews.append(modified_review)
                idx_list.append(idx)
            else:
                skipped_count += 1
    
    print(f"Skipped {skipped_count} reviews due to no suitable name replacement or token mismatch.")
    return original_reviews, modified_reviews, idx_list




class SentimentDataset(Dataset):
    def __init__(self, original, modified, ref_next_steps, cf_next_steps, classifier, category):
        cat_label = "positive" if category.startswith("pos") else "negative"
        original = [{"review": text, "gold_label": cat_label} for text in original]
        modified = [{"review": text, "gold_label": cat_label} for text in modified]

        # self.original = classifier.convert_to_features(original, "test")
        # self.modified = classifier.convert_to_features(modified, "test")

        self.original = classifier._process_dataset(original, "test")
        self.modified = classifier._process_dataset(modified, "test")
        
        self.ref_next_steps = ref_next_steps
        self.cf_next_steps = cf_next_steps
        self.tokenizer = classifier.tokenizer

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):
        return (self.original[idx], 
                self.modified[idx], 
                self.ref_next_steps[idx], 
                self.cf_next_steps[idx])

def collate_fn(batch, tokenizer):
    orig_texts, mod_texts, ref_next, cf_next = zip(*batch)
    
    orig_encoded = tokenizer.batch_encode_plus(
        [item['text'] for item in orig_texts],
        add_special_tokens=True,
        max_length=tokenizer.model_max_length,
        padding='longest',
        truncation=True,
        return_tensors="pt"
    )
    
    mod_encoded = tokenizer.batch_encode_plus(
        [item['text'] for item in mod_texts],
        add_special_tokens=True,
        max_length=tokenizer.model_max_length,
        padding='longest',
        truncation=True,
        return_tensors="pt"
    )
    
    return (
        orig_encoded['input_ids'], 
        mod_encoded['input_ids'], 
        torch.tensor(ref_next), 
        torch.tensor(cf_next)
    )



def batched_path_patching_step_N(classifier, original, modified, ref_next_steps, cf_next_steps, batch_size=8, metric_type="logit_diff", 
                                 per_example_results=False, num_workers=2, category=None, intermediate_nodes=False, patch_type = "v"):
    
    
    device = next(classifier.model.parameters()).device
    
    # Create dataset and dataloader
    dataset = SentimentDataset(original, modified, ref_next_steps, cf_next_steps, classifier, category)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=lambda b: collate_fn(b, classifier.tokenizer)
    )
    
    results = []
    batch_times = []

    # Process intermediate nodes 
    if intermediate_nodes is not False:
        print("intermediate_nodes - patch_type: ", patch_type)
        if patch_type == "mlp_out":
            print("PATCHING VIA mlp_out")
            patch_type = "resid_mid"
            receiver_nodes=[Node(patch_type, layer) for layer in intermediate_nodes]
            highest_layer = max([layer for layer in intermediate_nodes])
        else:
            receiver_nodes=[Node(patch_type, layer, head=head) for (layer, head) in intermediate_nodes]
            highest_layer = max([layer for layer, _ in intermediate_nodes])
        print('highest_layer', highest_layer)
    else:
        receiver_nodes=Node('resid_post', len(classifier.model.blocks)-1)
    
    # Compute reference and counterfactual logit differences
    print("Computing reference and counterfactual logit differences...")
    ref_logits_diff_list, cf_logits_diff_list = [], []
    for orig_ids, mod_ids, ref_next_batch, cf_next_batch in tqdm(dataloader, desc="Logit Diff Computation"):
        orig_ids, mod_ids, ref_next_batch, cf_next_batch = [t.to(device) for t in (orig_ids, mod_ids, ref_next_batch, cf_next_batch)]
        
        ref_logits = classifier.model(orig_ids, return_type='logits', prepend_bos=False)
        cf_logits = classifier.model(mod_ids, return_type='logits', prepend_bos=False)
        
        # ref_logit_diff = first_step_logit_diff(ref_logits, ref_next_batch, cf_next_batch, return_mean=True)
        # cf_logit_diff = first_step_logit_diff(cf_logits, ref_next_batch, cf_next_batch, return_mean=True)
        
        # ref_logits_diff_list.append(ref_logit_diff)
        # cf_logits_diff_list.append(cf_logit_diff)
        ref_logit_diff = first_step_logit_diff(ref_logits, ref_next_batch, cf_next_batch, return_mean=False)
        cf_logit_diff = first_step_logit_diff(cf_logits, ref_next_batch, cf_next_batch, return_mean=False)
        
        ref_logits_diff_list.append(ref_logit_diff.detach().cpu().numpy())
        cf_logits_diff_list.append(cf_logit_diff.detach().cpu().numpy())
    # c = [item for sublist in [a,b] for item in sublist]

    ref_logits_diff_list = [item for sublist in ref_logits_diff_list for item in sublist] 
    # ref_logit_diff = torch.mean(torch.stack(ref_logits_diff_list))
    ref_logit_diff = torch.mean(torch.Tensor(ref_logits_diff_list))
    # cf_logit_diff = torch.mean(torch.stack(cf_logits_diff_list))
    cf_logits_diff_list = [item for sublist in cf_logits_diff_list for item in sublist]
    cf_logit_diff = torch.mean(torch.Tensor(cf_logits_diff_list))

    print(f"Final logit diffs - Reference: {ref_logit_diff:.4f}, Counterfactual: {cf_logit_diff:.4f}")
    
    # Path patching
    print("Performing path patching...")
    for batch_idx, (orig_ids, mod_ids, ref_next_batch, cf_next_batch) in enumerate(tqdm(dataloader, desc="Path Patching")):
        batch_start_time = time.time()
        
        orig_ids, mod_ids, ref_next_batch, cf_next_batch = [t.to(device) for t in (orig_ids, mod_ids, ref_next_batch, cf_next_batch)]
        
        if metric_type == "logit_diff":
            metric_partial = partial(first_step_logit_diff, ref_next_steps=ref_next_batch, cf_next_steps=cf_next_batch)
        elif metric_type == "denoise":
            metric_partial = partial(first_step_metric_denoise, cf_logit_diff=cf_logit_diff, ref_logit_diff=ref_logit_diff,
                                     ref_next_steps=ref_next_batch, cf_next_steps=cf_next_batch)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        batch_results = path_patch(
            classifier.model,
            orig_input=orig_ids,
            new_input=mod_ids, 
            sender_nodes=IterNode(['z', 'mlp_out']),
            # sender_nodes=IterNode('z'),
            receiver_nodes=receiver_nodes,
            patching_metric=metric_partial,
            verbose=False,
            per_example_results=per_example_results,
        )
        results.append(batch_results)
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_times.append(batch_time)
        
        # Estimate remaining time
        mean_batch_time = np.mean(batch_times)
        remaining_batches = len(dataloader) - (batch_idx + 1)
        estimated_time_left = mean_batch_time * remaining_batches
        minutes, seconds = divmod(estimated_time_left, 60)
        print(f"Batch {batch_idx+1}/{len(dataloader)} complete. Estimated time left: {int(minutes)}m {int(seconds)}s")
    
    # Combine results from all batches
    if per_example_results:
        combined_results = {key: torch.cat([batch[key] for batch in results]) for key in results[0].keys()}
    else:
        combined_results = {key: torch.stack([batch[key] for batch in results]).mean(dim=0) for key in results[0].keys()}
    
    return combined_results, ref_logit_diff, cf_logit_diff


def visualize_results(results, ref_logits, ref_next_steps, cf_next_steps, model):
    if isinstance(results['z'], list):  # per_example_results=True
        show_logit_diff_heatmap_grid(results, ref_logits, ref_next_steps, cf_next_steps)
    else:
        results = results['z']
        model_config = model.config if hasattr(model, 'config') else model.cfg
        n_layers = model_config.n_layers
        n_heads = model_config.n_heads
        
        results = results.cpu().mean(dim=0) if results.dim() > 2 else results.cpu()
        results = results.reshape(n_layers, n_heads)
        ref_logit_diff = first_step_logit_diff(ref_logits, ref_next_steps, cf_next_steps, return_mean=True)
        cf_logit_diff = first_step_logit_diff(ref_logits, ref_next_steps, cf_next_steps, return_mean=True)
        results = (results - ref_logit_diff.cpu()) / (ref_logit_diff.cpu() - cf_logit_diff.cpu())
        
        fig = imshow_tensor_vis(
            results,
            title="Direct effect on logit diff (patch from head output -> final resid)",
            labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
            border=True,
            width=600,
            margin={"r": 100, "l": 100},
            return_fig=True
        )
        fig.show()

def main_patching_func(val_recast, classifier, batch_size=8, category='pos_good', metric_type="logit_diff", per_example_results=False, intermediate_nodes=False, max_samples=48, patch_type="v"):
    global first_names_male, first_names_female, last_names
    
    first_names_male, first_names_female, last_names = load_or_create_names()
    
    original, modified, idx_list = modify_reviews(val_recast, category)
    

    original = original[:max_samples]
    modified = modified[:max_samples]
    idx_list = idx_list[:max_samples]
    print("Number of samples used per category:", max_samples)
    print(f"Total number of reviews: {len(original)}")

    if category.startswith("pos"):
        ref_class_str = " B" # positive sentiment
        cf_class_str = " A" # negative sentiment
    else:
        ref_class_str = " A"
        cf_class_str = " B"

    ref_class = tokenizer.encode(ref_class_str, return_tensors="pt")[0].repeat(len(original)).tolist()
    cf_class = tokenizer.encode(cf_class_str, return_tensors="pt")[0].repeat(len(original)).tolist()    

    results, ref_logit_diff, cf_logit_diff = batched_path_patching_step_N(
        classifier,
        original,
        modified,
        ref_class,
        cf_class,
        batch_size=batch_size,
        metric_type="logit_diff",
        per_example_results=True,
        num_workers=2, 
        category=category,
        intermediate_nodes=intermediate_nodes,
        patch_type = patch_type
    )
    
    ref_cf_dict = {
        "ref_class": ref_class,
        "cf_class": cf_class,
        "ref_logit_diff": ref_logit_diff,
        "cf_logit_diff": cf_logit_diff
    }
    
    return results, original, modified, idx_list, ref_cf_dict



def load_trained_model(config_path, model_path, dataset_path, return_test_recast=False):
    # Load the configuration
    args = parse_config(config_path)

    logger = get_logger()
    
    # Initialize the GPTClassifier with the loaded configuration
    dataset_config = get_dataset_config(args.dataset)

    classifier = GPTClassifier(args, logger, dataset_config)
    # classifier = GPTClassifier(args, logger)
    state_dict = torch.load(model_path)
    classifier.model.load_state_dict(state_dict)
    
    # Load the dataset
    with open(dataset_path, 'rb') as f:
        train_recast, val_recast, test_recast, label_mapping = pickle.load(f)


    classifier.val_data_full = val_recast
    classifier.label_mapping = label_mapping


    classifier.model.eval()
    
    classifier.model.to("cuda")

    if return_test_recast:
        return classifier, val_recast, test_recast
    else:
        return classifier, val_recast
    


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Path Patch Batch Script")
    parser.add_argument("--max_samples", type=int, default=24, help="Maximum number of samples")
    parser.add_argument("--intermediate", type=str, default="True", choices=["True", "False"], help="Use intermediate nodes")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for processing")
    parser.add_argument("--patch_type", type=str, choices=["v", "k", "mlp_out"], default="mlp_out", help="Type of patching to perform (v or k)")

    args = parser.parse_args()

    ##############################
    ##### Parameters to Set ######
    exp_name = "SCperc_v2_WBIDmilvmakz"
    # exp_name = "SCperc_v2_WBID0yt9x5gg"
    # exp_name = "SCperc_v2_WBID1xijoxpb"


    use_intermediate = args.intermediate == "True"
    max_samples = args.max_samples
    batch_size = args.batch_size
    patch_type = args.patch_type

    print("max samples:", max_samples, ", Use intermed:", use_intermediate)

    ##############################
    ##### Parameters to Set ######
    intermediate_nodes_good = [(11, 2), (10, 0), (10,6)]
    intermediate_nodes_bad = [(11, 2), (10, 0), (10,6)]
    # intermediate_nodes_good= intermediate_nodes_bad = [0]

    ##############################

    result_path = "./results/" + exp_name + "/"
    config_path = result_path + "config.yml"
    model_path = result_path + "gpt2_imdb_classifier.pth"  # Update this path to where your model is saved
    dataset_path = result_path + "processed_imdb_dataset.pkl"  # Update this path to where your dataset is saved
    
    # Result filename 
    intermediate_suffix = f"_intermediate_{patch_type}" if use_intermediate else ""    
    result_file_name = f"{result_path}patching_results_dict{intermediate_suffix}.pkl"
    print("Stored results in:", result_file_name)

    classifier, val_recast = load_trained_model(config_path, model_path, dataset_path)


    # all_categories = ['pos_good', 'pos_bad', 'neg_good', 'neg_bad']
    all_categories = ['pos_bad', 'neg_good']


    final_results = {}
    for cat in all_categories:
        print("\n", "="*50)
        print("Processing category:", cat)
        intermediate_nodes = False
        if use_intermediate:
            if cat.endswith("good"):
                print("This is a Good category", cat)
                intermediate_nodes = intermediate_nodes_good
            else:
                print("This is a Bad category", cat)
                intermediate_nodes = intermediate_nodes_bad

        # Assuming val_recast and model are already loaded
        results, original, modified, idx_list, ref_cf_dict = main_patching_func(val_recast, classifier, batch_size=batch_size, category=cat, metric_type="logit_diff", per_example_results=True, 
                                                                  intermediate_nodes=intermediate_nodes, max_samples=max_samples, patch_type=patch_type)
        print("\nPath patching completed. Results, original reviews, modified reviews, and index list are available.")

        # put it all in one dict
        results_dict_cat = {
            "results": results,
            "original": original,
            "modified": modified,
            "idx_list": idx_list,
            "ref_cf_dict": ref_cf_dict
        }
        final_results[cat] = results_dict_cat



    with open(result_file_name, 'wb') as f:
        pickle.dump(final_results, f)