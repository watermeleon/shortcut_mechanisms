
import torch

# Local application/library specific imports
from transformer_lens import HookedTransformer, ActivationCache

from robin_nlp.mechinterp.logit_diff_functions import  *




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




