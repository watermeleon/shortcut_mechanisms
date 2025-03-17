
import torch

# Local application/library specific imports
from transformer_lens import HookedTransformer, ActivationCache

from robin_nlp.mechinterp.logit_backtrack_functions import *
from robin_nlp.mechinterp.logit_diff_functions import  *

from torch.utils.data import DataLoader
from tqdm import tqdm



def get_logit_diff_per_head_full(eval_sc_dataloader: DataLoader, model: HookedTransformer, answer_tokens: torch.Tensor, logit_diff_thresh=0.5, att_threshold=0.7):

    logit_diff_res_full = []
    for i, batch in enumerate(tqdm(eval_sc_dataloader)):
        input_ids, _, _, name_mask, _, _, _ = batch
        name_mask = name_mask.squeeze()

        logit_score_list, tok_mask_list = aggr_logitdiff_methodV2(model, input_ids, answer_tokens, logit_diff_thresh, att_threshold, return_individual_scores=True)
        logit_diff_res_full.append((logit_score_list, tok_mask_list))

    return logit_diff_res_full



def att_score_from_idx_v2(model: HookedTransformer, cache: ActivationCache, receiver_tuple, input_activations, answer_tokens,  apply_ln=False, maskout_low=False, softmax_attention=False, filt_ld_neg=False, att_times_ov_vals=True, add_bias=False):
    layer, head, token_idx = receiver_tuple
    if apply_ln:
        input_activations = cache.apply_ln_to_stack(input_activations, layer=layer, pos_slice=token_idx)

    # Load the q values from cache and compute k values from the decomposed residual
    Wk = model.W_K[layer, head]
    Wq_out = cache[f'blocks.{layer}.attn.hook_q'].squeeze()[token_idx, head]
    Wk_out = input_activations.squeeze() @ Wk 
    if add_bias:
        Wk_out = Wk_out + model.b_K[layer, head]

    att_score = Wk_out @ Wq_out  # shape: num tokens
    if softmax_attention:
        att_score = torch.nn.functional.softmax(att_score, dim=-1)

    # Get the OV transformed vectors per input
    W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
    OV_vals =  input_activations.squeeze() @ W_OV # shape num_tokens x d_model 

    if add_bias:
        bias_term = model.b_V[layer, head] @ model.W_O[layer, head] + model.b_O[layer, head]
        OV_vals = OV_vals + bias_term

    if att_times_ov_vals:
        OV_vals = OV_vals * att_score.unsqueeze(-1)

    # Get the logit diff from the OV values and Scale by the attention score
    logit_val =  get_logit_diff_from_activations(model, answer_tokens, OV_vals)

    if filt_ld_neg:
        # set all positive values to zero
        logit_val[logit_val > 0] = 0

    if maskout_low:
        bin_mask = get_top_mask_v2(logit_val, threshold=0.8)
        logit_val = logit_val * bin_mask
        return input_activations, att_score, logit_val, OV_vals, bin_mask

    
    return input_activations, att_score, logit_val, OV_vals.squeeze()


def get_attention_input(cache:ActivationCache, layer, token_idx, token_idx_looksat, from_input_embs):
    """
    Get the appropriate input vectors for the attention head analysis.
    
    Args:
        cache: Model cache containing intermediate values
        layer: Current layer number
        token_idx: Token index being analyzed
        token_idx_looksat: Token index being looked at
        from_input_embs: Boolean indicating whether to use input embeddings
        
    Returns:
        tuple: (attention head input tensor, token labels)
    """
    if from_input_embs:
        token_idx_temp = token_idx + 1 if token_idx >= 0 else token_idx
        att_head_input = cache[f'blocks.{layer}.ln1.hook_normalized'][:,:token_idx_temp]
        labels = None  # Labels will be provided by caller
    else:
        # stack_head_res will do this too otherwise, but will give an annoying error    
        if "blocks.0.attn.hook_result" not in cache.cache_dict:
            cache.compute_head_results()

        att_head_input, labels = cache.stack_head_results(layer, pos_slice=token_idx_looksat, 
                                                        return_labels=True, apply_ln=False)
        att_head_input, labels = add_input_embs(cache, att_head_input, token_idx_looksat, labels)
    
    return att_head_input.squeeze().unsqueeze(0), labels

def process_attention_scores(model, cache, answer_tokens, current_node, att_head_input, is_first_step):
    """
    Calculate and process attention scores for the current node.
    
    Args:
        model: Transformer model
        cache: Model cache
        current_node: Current Node object being analyzed
        att_head_input: Input tensor for attention calculation
        is_first_step: Boolean indicating if this is the first step
        
    Returns:
        tuple: (best attention index, attention scores, logit values)
    """
    receiver_tuple = (current_node.layer_id, current_node.head_id, current_node.token_idx)
    
    _, att_score, logit_val, _ = att_score_from_idx_v2(
        model, cache, receiver_tuple, att_head_input, 
        apply_ln=True, maskout_low=False, softmax_attention=True,
        filt_ld_neg=False, att_times_ov_vals=is_first_step, add_bias=True, 
        answer_tokens=answer_tokens
    )
    
    best_att_idx = torch.argmax(att_score).item()
    return best_att_idx, att_score, logit_val

def create_child_node(current_node, best_att_head):
    """
    Create a new child node based on the best attention head.
    
    Args:
        current_node: Current Node object
        best_att_head: String representing the best attention head (e.g., "L2H3")
        
    Returns:
        Node: New child node
    """
    source_l, source_h = best_att_head[1:].split("H")
    source_l, source_h = int(source_l), int(source_h)
    
    new_node = Node(
        layer_id=source_l,
        head_id=source_h,
        token_idx=current_node.token_idx_looksat_id,
        token_idx_looksat_id=None
    )
    current_node.add_child(new_node)
    return new_node



def explore_attention_path(current_node, cache, answer_tokens, label_toks, model, depth=0, max_depth=5, threshold=0.3):
    """
    Recursively explore attention paths where attention score > threshold.
    
    Args:
        current_node: Current Node object being explored
        cache: Model cache containing intermediate values
        label_toks: Token labels
        model: Transformer model
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops
        threshold: Minimum attention score to follow (default 0.3 = 30%)
    
    Returns:
        list: Collection of results for this path
    """
    if depth >= max_depth:
        return []
    
    results = []
    
    # First step: Look at input embeddings
    att_head_input, _ = get_attention_input(
        cache, current_node.layer_id, current_node.token_idx,
        current_node.token_idx_looksat_id, from_input_embs=True
    )
    
    # Get attention scores for input embeddings of current head
    _, att_score, logit_val = process_attention_scores(
        model, cache, answer_tokens, current_node, att_head_input, is_first_step=True, 
    )
    
    # Find all positions with attention score > threshold
    max_score = torch.max(att_score) * threshold
    high_attention_indices = torch.where(att_score >= max_score )[0]

    # If no high attention scores found, this branch ends
    if len(high_attention_indices) == 0:
        return results
    
    # Update current node with all high-attention positions it looks at
    for att_idx in high_attention_indices:
        current_node.token_idx_looksat_id = att_idx.item()
        
        # Now look at previous layer heads
        att_head_input, labels = get_attention_input(
            cache, current_node.layer_id, current_node.token_idx,
            current_node.token_idx_looksat_id, from_input_embs=False
        )
        
        # Get attention scores for previous layer heads
        _, att_score, logit_val = process_attention_scores(
            model, cache, answer_tokens, current_node, att_head_input, is_first_step=False
        )
        
        # Find all heads with high attention
        high_attention_heads = torch.tensor([torch.argmax(att_score)])


        for head_idx in high_attention_heads:
            best_att_head = labels[head_idx]
            
            # Check if we've reached input embeddings
            if best_att_head == 'input_embs':
                # Create final input embedding node
                final_node = Node(
                    layer_id=-1,
                    head_id="input_emb",
                    token_idx=current_node.token_idx_looksat_id,
                    token_idx_looksat_id=0
                )
                current_node.add_child(final_node)

                if att_score.dim() != 0:
                    # If we are at the first layer (from the input) then there is only 1 option, otherwise we do need to take the head index
                    att_score = att_score[head_idx]
                    
                results.append((
                    -1, 
                    "input_emb", 
                    current_node.token_idx_looksat_id, 
                    att_score.item(), 
                    logit_val, 
                    labels
                ))
                continue
                
            # Process regular attention head
            new_node = create_child_node(current_node, best_att_head)
            
            # Store current step results
            results.append((
                new_node.layer_id, 
                new_node.head_id, 
                new_node.token_idx, 
                att_score[head_idx].item(), 
                logit_val, 
                labels
            ))        
            
            # Base case: reached input embeddings
            if new_node.layer_id == -1:
                if new_node.token_idx_looksat_id is not None:
                    final_node = Node(
                        layer_id=-1,
                        head_id="pos_emb",
                        token_idx=new_node.token_idx_looksat_id,
                        token_idx_looksat_id=0
                    )
                    new_node.add_child(final_node)
                continue
            
            # Recursive call to explore this path
            sub_results = explore_attention_path(
                new_node, cache, answer_tokens,  label_toks, model,
                depth + 1, max_depth, threshold
            )
            results.extend(sub_results)
    
    return results

def backtrack_heads(model, answer_tokens, start_head, input_tokens, threshold=0.3, max_depth=5):
    """
    Main function to recursively backtrack attention heads through the transformer network.
    Explores all paths where attention score > threshold.
    
    Args:
        start_head: Tuple of (layer, head, token_idx, token_idx_looksat)
        input_tokens: Input token tensor
        threshold: Minimum attention score to follow (default 0.3 = 30%)
        max_depth: Maximum recursion depth
        
    Returns:
        tuple: (root Node, token labels, full results list)
    """
    _, cache = model.run_with_cache(input_tokens, prepend_bos=True)
    label_toks = model.to_str_tokens(input_tokens)

    # Create root node
    layer, head, token_idx, token_idx_looksat = start_head
    root = Node(layer_id=layer, head_id=head, token_idx=token_idx, token_idx_looksat_id=token_idx_looksat)
    
    # Start recursive exploration
    full_res = explore_attention_path(
        root, cache, answer_tokens, label_toks, model,
        depth=0, max_depth=max_depth, threshold=threshold
    )
    
    return root, label_toks, full_res


def aggr_logitdiff_methodV2(model, input_tokens, answer_tokens, logit_diff_thresh=0.5, att_threshold=0.7, return_individual_scores=False):
    """ Aggregate logit difference scores for the input sentence:
            Inspect the attention heads with the most significant logit difference scores
            Backtrack the heads and attribute each token with the logit difference score of the label original head"""	
    keep_mask, per_head_logit_diffs, _ = get_head_skip_mask(model, input_tokens, answer_tokens, logit_diff_thresh)

    input_tokens = input_tokens.squeeze()
    logit_scores = torch.zeros_like(input_tokens).float()

    logit_score_list = []
    tok_mask_list = []

    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            if not keep_mask[i, j]:
                continue

            head_logit_diff = per_head_logit_diffs[i, j]

            start_head = (i, j, len(input_tokens) - 1, None)  # head is: layer_id, head_id, token_idx, token_idx_looksat_id

            root_node, _, _ = backtrack_heads(model, answer_tokens, start_head, input_tokens, att_threshold)
            used_toks_mask = root_node.get_input_tokens_mask(sequence_length=len(input_tokens))

            logit_scores[used_toks_mask] += head_logit_diff

            tok_mask_list.append(used_toks_mask)
            logit_score_list.append(head_logit_diff)


    if return_individual_scores:
        return logit_score_list, tok_mask_list
    else:
        return logit_scores



def aggr_logitdiff_methodV2_2_filter_names(model, input_tokens, answer_tokens, name_mask, review_mask, logit_diff_thresh=0.5, att_threshold=0.7):
    """ V2.2: Filter logit_diff heads that attribute the result to more than the Actor Name tokens. """
    logit_score_list, tok_mask_list = aggr_logitdiff_methodV2(model, input_tokens, answer_tokens, logit_diff_thresh, att_threshold, return_individual_scores=True)

    input_tokens = input_tokens.squeeze()
    device = next(model.parameters()).device
    
    logit_scores = torch.zeros_like(input_tokens).float()
    for i in range(len(tok_mask_list)):
        used_toks_mask = tok_mask_list[i].to(device)
        head_logit_diff = logit_score_list[i]

        used_non_name_review_mask = review_mask & ~name_mask & used_toks_mask
        if used_non_name_review_mask.sum() > 0:
            continue
        
        logit_scores[used_toks_mask] += head_logit_diff

    return logit_scores

