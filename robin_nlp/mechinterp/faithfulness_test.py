
import pandas as pd
from IPython.display import display

# Third-party imports
import torch
from tqdm import tqdm
import random

torch.set_grad_enabled(False)

# Local application/library specific imports

from transformer_lens import HookedTransformer
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier
from robin_nlp.data.imdb_helper_functions import *

from robin_nlp.mechinterp.logit_diff_functions import *
from robin_nlp.mechinterp.visualizations import *
from robin_nlp.mechinterp.path_patch_batch import modify_reviews
from robin_nlp.mechinterp.faithfulness_helper import *
from robin_nlp.mechinterp.path_patching import Node, IterNode

from robin_nlp.notebook_utils import *

import argparse


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
        processed_samples["template_idx"] = template_idx
        processed_samples["ref_name"] = ref_name
        processed_samples["cf_name"] = cf_name

        processed_samples_list.append(processed_samples)
    
    return processed_samples_list, ref_prompts, cf_prompts, idx_list


def get_patching_data_batched(classifier: GPTClassifier, tokenizer, ref_prompts, cf_prompts, cat_label, batch_size=32, prompts_and_labeltoks_only=False):
    """
    Process and return patching data in batches for memory efficiency.
    
    Args:
        classifier: GPTClassifier instance
        tokenizer: Tokenizer for processing text
        ref_prompts: List of reference prompts
        cf_prompts: List of counterfactual prompts
        cat_label: Category label
        batch_size: Size of each batch
        prompts_and_labeltoks_only: Flag to return only prompts and label tokens
        
    Returns:
        Tuple of lists, where each list contains batched tensors
    """
    model: HookedTransformer = classifier.model
    device = next(classifier.model.parameters()).device
    num_samples = len(ref_prompts)
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Initialize lists to store batched data
    ref_prompts_toks_batches = []
    ref_prompts_mask_batches = []
    cf_prompts_toks_batches = []
    cf_prompts_mask_batches = []
    ref_next_steps_batches = []
    cf_next_steps_batches = []

    # Process data in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        batch_size_current = end_idx - start_idx

        # Format and encode reference prompts for current batch
        ref_batch = [classifier.format_prompt(prompt, cat_label, return_string=True)[0] 
                    for prompt in ref_prompts[start_idx:end_idx]]
        ref_encoded = classifier.encode_batch(ref_batch)
        ref_prompts_toks_batches.append(ref_encoded['input_ids'].to(device))
        ref_prompts_mask_batches.append(ref_encoded['attention_mask'].to(device))

        # Format and encode counterfactual prompts for current batch
        cf_batch = [classifier.format_prompt(prompt, cat_label, return_string=True)[0] 
                   for prompt in cf_prompts[start_idx:end_idx]]
        cf_encoded = classifier.encode_batch(cf_batch)
        cf_prompts_toks_batches.append(cf_encoded['input_ids'].to(device))
        cf_prompts_mask_batches.append(cf_encoded['attention_mask'].to(device))

        # Generate label tokens for current batch
        ref_next_step_batch = tokenizer.encode(" A", return_tensors="pt")[0].repeat(batch_size_current)
        cf_next_step_batch = tokenizer.encode(" B", return_tensors="pt")[0].repeat(batch_size_current)
        ref_next_steps_batches.append(ref_next_step_batch)
        cf_next_steps_batches.append(cf_next_step_batch)

    return (ref_prompts_toks_batches, ref_prompts_mask_batches, 
            cf_prompts_toks_batches, cf_prompts_mask_batches, 
            ref_next_steps_batches, cf_next_steps_batches)


def prepare_patching_data_faith2(classifier: GPTClassifier, num_samples, original, modified, idx_list_mod, test_recast, cat_label, batch_size=32):
    """
    Prepare batched data for path patching by generating next steps, tokenizing prompts, and extracting name indices.
    
    Args:
        classifier: GPTClassifier instance
        num_samples: Number of samples to process
        original: Original prompts
        modified: Modified prompts
        idx_list_mod: List of indices for modifications
        test_recast: Test recasting flag
        cat_label: Category label
        batch_size: Size of each batch
        
    Returns:
        Tuple containing batched tensors and indices for reference prompts, counterfactual prompts,
        next steps, and name indices
    """
    processed_samples_list, ref_prompts, cf_prompts, _ = process_samples(num_samples, original, modified, idx_list_mod, test_recast)

    # Get batched patching data
    patching_data = get_patching_data_batched(
        classifier, 
        tokenizer, 
        ref_prompts, 
        cf_prompts, 
        cat_label=cat_label, 
        batch_size=batch_size, 
        prompts_and_labeltoks_only=True
    )
    
    ref_prompts_toks_batches, ref_prompts_mask_batches, cf_prompts_toks_batches, cf_prompts_mask_batches, ref_next_steps_batches, cf_next_steps_batches = patching_data

    # Extract names and process them in batches
    names = [processed_samples_list[i]["ref_name"] for i in range(len(processed_samples_list))]
    ref_name_ind_batches = []
    
    for batch_idx, ref_prompts_toks_batch in enumerate(ref_prompts_toks_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(names))
        names_batch = names[start_idx:end_idx]
        processed_samples_batch = processed_samples_list[start_idx:end_idx]
        
        ref_name_ind_batch = get_list_of_name_indx(
            names_batch, 
            ref_prompts_toks_batch, 
            classifier.model, 
            processed_samples_batch
        )
        ref_name_ind_batches.append(ref_name_ind_batch)

    return (ref_prompts_toks_batches, ref_prompts_mask_batches, 
            cf_prompts_toks_batches, cf_prompts_mask_batches, 
            ref_next_steps_batches, cf_next_steps_batches, 
            ref_name_ind_batches, processed_samples_list)



def get_probs_cf_ref(classifier: GPTClassifier, ref_prompts_toks_batches, ref_prompts_mask_batches, 
                    cf_prompts_toks_batches, cf_prompts_mask_batches):
   probs_rf_list = []
   probs_cf_list = []
   
   for batch_idx in tqdm(range(len(ref_prompts_toks_batches)), desc="Predicting probabilities"):
       ref_batch_probs = classifier.predict_batch(
           ref_prompts_toks_batches[batch_idx], 
           ref_prompts_mask_batches[batch_idx]
       )
       ref_batch_probs = torch.round(ref_batch_probs[:, 0] * 100, decimals=2)
       probs_rf_list.append(ref_batch_probs.detach().cpu().numpy())
       
       cf_batch_probs = classifier.predict_batch(
           cf_prompts_toks_batches[batch_idx], 
           cf_prompts_mask_batches[batch_idx]
       )
       cf_batch_probs = torch.round(cf_batch_probs[:, 0] * 100)
       probs_cf_list.append(cf_batch_probs.detach().cpu().numpy())
   
   return np.concatenate(probs_rf_list), np.concatenate(probs_cf_list)


def calculate_patching_errors(results_2, probs_rf, probs_cf, orig_is_ref=True, cat_label="pos_good"):
    data = {
        "Index": [],
        "Ref prob": [],
        "CF prob": [],
        "Patched": [],
        "Error was": [],
        "Error after patching": []
    }
    results_2_cpu = [round(res.cpu().item(), 2) for res in results_2]

    ref_correct_list = []
    cf_correct_list = []
    pp_correct_list = []

    def gt_50(val):
        return val > 50
    def lt_50(val):
        return val < 50
    
    correct_operator = gt_50 if cat_label.startswith("neg") else lt_50

    for i in range(len(results_2_cpu)):
        error_was = round(abs(probs_rf[i] - probs_cf[i]), 2)
        if orig_is_ref:
            error_after_patching = round(abs(results_2_cpu[i] - probs_cf[i]), 2)
        else:
            error_after_patching = round(abs(probs_rf[i] - results_2_cpu[i]), 2)

        data["Index"].append(i)
        data["Ref prob"].append(probs_rf[i])
        data["CF prob"].append(probs_cf[i])
        data["Patched"].append(results_2_cpu[i])
        data["Error was"].append(error_was)
        data["Error after patching"].append(abs(error_after_patching))

        pp_correct_list.append(correct_operator(results_2_cpu[i] ))
        ref_correct_list.append(correct_operator(probs_rf[i] ))
        cf_correct_list.append(correct_operator(probs_cf[i] ))


    df_results = pd.DataFrame(data)
    print(f" Number of samples: {len(df_results)}")
    print("\n### Difference in probabilities between CF and Refs:")
    print(f"- Old avg error:: {sum(df_results['Error was'])/len(df_results['Error was']):.2f}")
    print(f"- Patched avg error: {sum(df_results['Error after patching'])/len(df_results['Error after patching']):.2f}")

    # print accuracy per list
    print("\n### Accuracy of Ref, CF and Patched:")
    print(f"- Ref Accuracy: {sum(ref_correct_list)/len(ref_correct_list) * 100:.1f}%")
    print(f"- CF Accuracy: {sum(cf_correct_list)/len(cf_correct_list) * 100:.1f}%")
    print(f"- PP Accuracy: {sum(pp_correct_list)/len(pp_correct_list) * 100:.1f}%")

    display(df_results.head(10))
    return df_results


#  Check if file is called directly (main)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run faithfulness test with path patching.")
    parser.add_argument("--exp_name", type=str, default="SCperc_v2_WBIDmilvmakz", help="Experiment name")
    parser.add_argument("--cat_label", type=str, default="neg_bad", choices=["neg_good", "pos_good", "pos_bad", "neg_bad"], help="Category label")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size")

    args = parser.parse_args()

    exp_name = args.exp_name
    cat_label = args.cat_label
    num_samples = args.num_samples
    seed = args.seed
    batch_size = args.batch_size

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  

    use_batched = True

    result_folder = f'./results/{exp_name}/'
    dataset_filename = f'{result_folder}processed_imdb_dataset.pkl'

    train_data, val_data, test_data, label_mapping = load_pickle(dataset_filename)

    classifier, model, test_recast = load_trained_model(exp_name, result_path=result_folder, return_split="test")
    tokenizer = model.tokenizer

    answer_tokens = get_answer_tokens(model)

    original, modified, idx_list_mod = modify_reviews(test_recast, cat_label)
    num_samples = min(num_samples, len(original))
    print("Num samples is now:", num_samples)

    patching_data = prepare_patching_data_faith2(classifier, num_samples, original, modified, idx_list_mod, test_recast, cat_label)
    ref_prompts_toks, ref_prompts_mask, cf_prompts_toks, cf_prompts_mask, ref_next_steps, cf_next_steps, ref_name_ind, processed_samples_list = patching_data

    print("#### Number of samples: ", len(ref_prompts_toks))

    intermediate_nodes = [(10, 0), (10, 6), (11, 2)]

    sender_nodes = []
    for i in range (1):
        sender_nodes.append(Node("mlp_out", i))


    probs_rf, probs_cf = get_probs_cf_ref(classifier, ref_prompts_toks, ref_prompts_mask, cf_prompts_toks, cf_prompts_mask)

    # Patch via intermediate nodes:
    receiver_nodes = []
    for layer, head in intermediate_nodes:
        receiver_nodes.append(Node("v", layer, head=head))
        receiver_nodes.append(Node("k", layer, head=head))

    if use_batched:
        results_2 = apply_path_patching_with_intermediate_nodes_batched(model, ref_prompts_toks, cf_prompts_toks, 
                                              ref_next_steps, cf_next_steps, ref_name_ind, sender_nodes, receiver_nodes)
    else:
        results_2 = apply_path_patching_with_intermediate_nodes(model, ref_prompts_toks, cf_prompts_toks, ref_next_steps, cf_next_steps, processed_samples_list, sender_nodes, receiver_nodes)
    print("Finished Patching")

    df_results = calculate_patching_errors(results_2, probs_rf, probs_cf, orig_is_ref=False, cat_label= cat_label)