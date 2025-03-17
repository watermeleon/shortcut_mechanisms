
from jaxtyping import Float
from functools import partial

import torch
from tqdm import tqdm

from transformer_lens.hook_points import HookPoint  
from transformer_lens import utils, HookedTransformer, ActivationCache

import pandas as pd
from functools import partial
from tqdm import tqdm
from IPython.display import display

from collections import defaultdict
from typing import List, Dict
from copy import deepcopy

def get_reviews_cat(dataset: List[Dict]) -> Dict[str, List[str]]:
    dataset = deepcopy(dataset)

    review_per_cat = defaultdict(list)    

    for idx, item in tqdm(enumerate(dataset)):
        category = item['category']
        original_review = item['review']
        label = item['gold_label']

        # ensure all categories have pos_ or neg_ as a prefix
        if category == "clean_review":
            category = label[:3] + "_" + category

        review_per_cat[category].append(original_review)

    total_reviews = sum(len(reviews) for reviews in review_per_cat.values())

    # print the size of each category and its percentage of the total data
    for key, value in review_per_cat.items():
        percentage = (len(value) / total_reviews) * 100
        print(f"Category: {key}, Size: {len(value)}, Percentage: {percentage:.2f}%")

    return review_per_cat


def head_ablation_hook(
    value: Float[torch.Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index_to_ablate: int
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    value[:, :, head_index_to_ablate, :] = 0.
    return value


def mlp_ablation_hook(
    mlp_out: Float[torch.Tensor, "batch pos d_model"],
    hook: HookPoint,
    layer_to_ablate: int
) -> Float[torch.Tensor, "batch pos d_model"]:
    mlp_out[:] = 0.  # Zero out the entire MLP output
    return mlp_out

def get_correct_incorrect_class_tokens(category):
    if category.startswith("pos"):
        incorrect_class_tokenid = 317  # A 
        correct_class_tokenid = 347  # B
    else:
        incorrect_class_tokenid = 347
        correct_class_tokenid = 317

    return correct_class_tokenid, incorrect_class_tokenid


def abalate_one_att_head(category, layer_to_ablate, head_index_to_ablate, original, ref_class, cf_class, classifier, print_results=True):
    model = classifier.model
    
    cat_label = "positive" if category.startswith("pos") else "negative"

    og_dataloader = classifier.prepare_single_dataset(original, cat_label, batch_size=16)
    correct_class_tokenid, incorrect_class_tokenid = get_correct_incorrect_class_tokens(category)

    head_ablation_hook_partial = partial(head_ablation_hook, head_index_to_ablate=head_index_to_ablate)

    og_correct_pred_list = []
    correct_pred_list = []

    for batch in tqdm(og_dataloader, disable=True):
        input_ids, attention_mask, labels = batch

        original_logits = model(input_ids, attention_mask=attention_mask)

        og_corr_logits = original_logits[:, -2,  correct_class_tokenid]
        og_incorr_logits = original_logits[:, -2, incorrect_class_tokenid]

        ablated_logits = model.run_with_hooks(
            input_ids, 
            attention_mask=attention_mask,
            fwd_hooks=[(
                utils.get_act_name("v", layer_to_ablate), 
                head_ablation_hook_partial
                )]
            )
        
        ablated_corr_logits = ablated_logits[:, -2,  correct_class_tokenid]
        ablated_incorr_logits = ablated_logits[:, -2, incorrect_class_tokenid]

        correct_pred_list.extend((ablated_corr_logits > ablated_incorr_logits).tolist())
        og_correct_pred_list.extend((og_corr_logits > og_incorr_logits).tolist())

    accuracy = sum(correct_pred_list) / len(correct_pred_list) * 100
    acc_og = sum(og_correct_pred_list) / len(og_correct_pred_list) * 100

    if print_results:
        print(f"Head Index: {head_index_to_ablate}, at Layer {layer_to_ablate}")
        print(f"Average Accuracy: {accuracy:.2f}%, and before ablation it was {acc_og:.2f}%")
        print("-"*50)
    else:
        return accuracy, acc_og



def att_head_ablation_onelayer(category, original, cf_class, ref_class, classifier, layer_to_ablate=10, batch_size=16, print_results=True):
    print("Category is: ", category)
    model = classifier.model
    cat_label = "positive" if category.startswith("pos") else "negative"

    og_dataloader = classifier.prepare_single_dataset(original, cat_label, batch_size=batch_size)
  
    correct_class_tokenid, incorrect_class_tokenid = get_correct_incorrect_class_tokens(category)

    best_head = None
    best_accuracy = 0
    results = []
    
    for head_index_to_ablate in tqdm(range(12)):
        head_ablation_hook_partial = partial(head_ablation_hook, head_index_to_ablate=head_index_to_ablate)
        
        og_correct_pred_list = []
        correct_pred_list = []
        
        for batch in tqdm(og_dataloader, disable=True):
            input_ids, attention_mask, labels = batch
            original_logits = model(input_ids, attention_mask=attention_mask)
            og_corr_logits = original_logits[:, -2, correct_class_tokenid]
            og_incorr_logits = original_logits[:, -2, incorrect_class_tokenid]
            
            ablated_logits = model.run_with_hooks(
                input_ids,
                attention_mask=attention_mask,
                fwd_hooks=[(
                    utils.get_act_name("v", layer_to_ablate),
                    head_ablation_hook_partial
                )]
            )
           
            ablated_corr_logits = ablated_logits[:, -2, correct_class_tokenid]
            ablated_incorr_logits = ablated_logits[:, -2, incorrect_class_tokenid]
            correct_pred_list.extend((ablated_corr_logits > ablated_incorr_logits).tolist())
            og_correct_pred_list.extend((og_corr_logits > og_incorr_logits).tolist())
        
        accuracy = sum(correct_pred_list) / len(correct_pred_list) * 100
        acc_og = sum(og_correct_pred_list) / len(og_correct_pred_list) * 100
        
        results.append({
            "Layer": layer_to_ablate,
            "Head": head_index_to_ablate,
            "Ablated_Accuracy": round(accuracy, 2),
            "Original_Accuracy": round(acc_og, 2)
        })
        
        if print_results:
            print(f"Head Index: {head_index_to_ablate}, at Layer {layer_to_ablate}")
            print(f"Average Accuracy: {accuracy:.2f}%, and before ablation it was {acc_og:.2f}%")
            print("-"*50)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_head = head_index_to_ablate

    df = pd.DataFrame(results)
    df.set_index('Head', inplace=True)  # Set 'Head' as the index

    
    if not print_results:
        print("Results:")
        display(df)
    
    print("\n" + "="*50)
    print(f"Best Att Head: {best_head}, with Accuracy: {best_accuracy:.2f}%")
    print("="*50)

    return best_head, best_accuracy, df


def mlp_ablation_all_layers(category, original, classifier, print_results=True):
    print("Category is: ", category)
    model = classifier.model
 
    best_layer = None
    best_accuracy = 0
    results = []

    correct_class_tokenid, incorrect_class_tokenid = get_correct_incorrect_class_tokens(category)
    cat_label = "positive" if category.startswith("pos") else "negative"

    og_dataloader = classifier.prepare_single_dataset(original, cat_label)
    
    for layer_to_ablate in tqdm(range(12), desc="Layer"):
        for act_type in ["post"]:
            og_correct_pred_list = []
            correct_pred_list = []
            
            for batch in tqdm(og_dataloader, disable=True):
                input_ids, attention_mask, labels = batch
                original_logits = model(input_ids, attention_mask=attention_mask)
                og_corr_logits = original_logits[:, -2, correct_class_tokenid]
                og_incorr_logits = original_logits[:, -2, incorrect_class_tokenid]
                
                mlp_ablation_hook_partial = partial(mlp_ablation_hook, layer_to_ablate=layer_to_ablate)
                
                ablated_logits = model.run_with_hooks(
                    input_ids,
                    attention_mask=attention_mask,
                    fwd_hooks=[(
                        utils.get_act_name(act_type, layer_to_ablate), mlp_ablation_hook_partial)
                    ]
                )
               
                ablated_corr_logits = ablated_logits[:, -2, correct_class_tokenid]
                ablated_incorr_logits = ablated_logits[:, -2, incorrect_class_tokenid]
                correct_pred_list.extend((ablated_corr_logits > ablated_incorr_logits).tolist())
                og_correct_pred_list.extend((og_corr_logits > og_incorr_logits).tolist())
           
            accuracy = sum(correct_pred_list) / len(correct_pred_list) * 100
            acc_og = sum(og_correct_pred_list) / len(og_correct_pred_list) * 100
            
            results.append({
                "Layer": layer_to_ablate,
                "Ablated_Accuracy": round(accuracy, 2),
                "Original_Accuracy": round(acc_og, 2)
            })
            
            if print_results:
                print(f"Ablation of layer {layer_to_ablate}, type: {act_type}")
                print(f"Average Accuracy: {accuracy:.2f}%, and before ablation it was {acc_og:.2f}%")
                print("-"*50)
           
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer_to_ablate

    df = pd.DataFrame(results)
    df.set_index('Layer', inplace=True)  # Set 'Layer' as the index
    
    if not print_results:
        print("Results:")
        display(df)

    print("\n" + "="*50)
    print(f"Best Layer: {best_layer}, with Accuracy: {best_accuracy:.2f}%")
    print("="*50)

    return best_layer, best_accuracy, df



def ablation_full_dataset(classifier, review_per_cat, att_head_ablation_idx=None, mlp_ablation_idx=None, print_results=True):
    results = []
    total_correct_pred_list = []
    total_correct_pred_list_og = []

    for cat_label, cat_reviews in tqdm(review_per_cat.items()):
        if print_results:
            print(f"Evaluating Category {cat_label}")

        cat_label_recast = "positive" if cat_label.startswith("pos") else "negative"

        og_dataloader = classifier.prepare_single_dataset(cat_reviews, cat_label_recast, batch_size=16)

        correct_class_tokenid, incorrect_class_tokenid = get_correct_incorrect_class_tokens(cat_label)

        fwd_hooks = []
        if att_head_ablation_idx is not None:
            for layer, head in att_head_ablation_idx:
                head_ablation_hook_partial = partial(head_ablation_hook, head_index_to_ablate=head)
                fwd_hooks.append((utils.get_act_name("v", layer), head_ablation_hook_partial))
        elif mlp_ablation_idx is not None:
            for mlp_layer in mlp_ablation_idx:
                mlp_ablation_hook_partial = partial(mlp_ablation_hook, layer_to_ablate=mlp_layer)
                fwd_hooks.append((utils.get_act_name("post", mlp_layer), mlp_ablation_hook_partial))
                            

        og_correct_pred_list = []
        correct_pred_list = []

        for batch in tqdm(og_dataloader, disable=not print_results):
            input_ids, attention_mask, labels = batch

            original_logits = classifier.model(input_ids, attention_mask=attention_mask)

            og_corr_logits = original_logits[:, -2, correct_class_tokenid]
            og_incorr_logits = original_logits[:, -2, incorrect_class_tokenid]

            ablated_logits = classifier.model.run_with_hooks(
                input_ids, 
                attention_mask=attention_mask,
                fwd_hooks=fwd_hooks
            )
            
            ablated_corr_logits = ablated_logits[:, -2, correct_class_tokenid]
            ablated_incorr_logits = ablated_logits[:, -2, incorrect_class_tokenid]

            correct_pred_list.extend((ablated_corr_logits > ablated_incorr_logits).tolist())
            og_correct_pred_list.extend((og_corr_logits > og_incorr_logits).tolist())

        accuracy = sum(correct_pred_list) / len(correct_pred_list) * 100
        acc_og = sum(og_correct_pred_list) / len(og_correct_pred_list) * 100

        results.append({
            "Category": cat_label,
            "Ablated_Accuracy": round(accuracy,2),
            "Original_Accuracy": round(acc_og, 2),
            "Num_Samples": len(correct_pred_list),
            "Ablation impacts": round(accuracy - acc_og, 2)
        })

        total_correct_pred_list.extend(correct_pred_list)
        total_correct_pred_list_og.extend(og_correct_pred_list)

        if print_results:
            print(f"Attention Heads to ablate (layer, head): {str(att_head_ablation_idx)}, MLP layers to ablate: {str(mlp_ablation_idx)}")
            print(f"Average Accuracy: {accuracy:.2f}%, and before ablation it was {acc_og:.2f}%")
            print("-" * 50)

    df = pd.DataFrame(results)
    
    total_accuracy = sum(total_correct_pred_list) / len(total_correct_pred_list) * 100
    total_accuracy_og = sum(total_correct_pred_list_og) / len(total_correct_pred_list_og) * 100
    total_samples = len(total_correct_pred_list)

    if not print_results:
        print("Results per category:")
        display(df)

    print("=" * 50)
    print(f"Total Average Accuracy: {total_accuracy:.2f}%")
    print(f"Total Average Accuracy Before Ablation: {total_accuracy_og:.2f}%")
    print(f"Total Number of Samples: {total_samples}")
    print("=" * 50)

    return df, total_correct_pred_list, total_correct_pred_list_og