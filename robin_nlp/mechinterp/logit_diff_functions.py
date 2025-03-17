from IPython.display import display

# Third-party imports
import torch
import matplotlib.pyplot as plt


from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
from torch import Tensor
import einops
import torch
import matplotlib.pyplot as plt

from jaxtyping import Float

from transformer_lens import utils, HookedTransformer, ActivationCache
from robin_nlp.mechinterp.visualizations import *


def get_logit_diff_direction(model, answer_tokens):
    """ Get logit diff vector for: answer tokens (correct, incorrect)"""
        
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens) # [batch 2 d_model]

    correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
    logit_diff_directions = correct_residual_directions - incorrect_residual_directions # [batch d_model]
    return logit_diff_directions




def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"] ,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    '''
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size


def get_per_head_logit_diffs(model: HookedTransformer, input_tokens, answer_tokens, return_cache=False):
    _, cache = model.run_with_cache(input_tokens)
    cache.compute_head_results() # stack_head_res will do this too otherwise, but will give an annoying error
    per_head_residual, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    per_head_residual = einops.rearrange(
        per_head_residual,
        "(layer head) ... -> layer head ...",
        layer=model.cfg.n_layers
    )

    logit_diff_directions = get_logit_diff_direction(model, answer_tokens)

    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache, logit_diff_directions = logit_diff_directions)
    
    if return_cache is True:
        return per_head_logit_diffs, cache
    else:
        return per_head_logit_diffs
    

def get_logit_diff_direction(model, answer_tokens):
    """ Get logit diff vector for: answer tokens (correct, incorrect)"""
        
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens) # [batch 2 d_model]

    correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
    logit_diff_directions = correct_residual_directions - incorrect_residual_directions # [batch d_model]
    return logit_diff_directions


def get_logit_diff_from_activations(model, answer_tokens, activations):
    """ Get logit diff for a given set of activations
             - differs from residual_stack_to_logit_diff() : no cache +  no mean over batch
    """
    ln_final: LayerNorm = model.ln_final
    
    ln_resid = ln_final(activations)

    logit_diff_directions = get_logit_diff_direction(model, answer_tokens)

    # logit_diff_directions
    logit_diff = ln_resid @ logit_diff_directions.T
    return logit_diff.squeeze()
    


def compare_attention_scores(att_pattern, attention_scores):
    # check each sums to 1
    print("att_pattern sum:", att_pattern.sum())
    print("attention_scores sum:", attention_scores.sum())

    # plot the two attention scores using plt
    plt.figure(figsize=(6, 3))
    plt.plot(attention_scores.detach().cpu(), lw=3, label="Calculated Attention Scores")
    plt.plot(att_pattern.detach().cpu(), lw=3, linestyle='dashed',  label="Stored Attention Pattern")
    plt.xlabel('Token Position')
    plt.ylabel('Attention Score')
    plt.title('Attention Scores Comparison')
    plt.legend()
    plt.show()
    
def add_input_embs(cache, resid_tens, pos_idx, labels):
    """ Add the input embeddings to the beginning of the residual tensor and labels """
    # resid_tens is shape:  n_heads, 1 , d_model
    # labels is shape:  n_heads
    input_embs = cache['blocks.0.hook_resid_pre'][:,pos_idx].unsqueeze(0) # shape 

    new_resid = torch.cat([input_embs, resid_tens], dim=0)

    labels = ["input_embs"] + labels

    return new_resid, labels


def get_top_mask_v2(values, threshold=0.8):
    """
    Create a binary mask for the Top X% of a probability distribution.
        - Get the indices from bottom and invert (bit complicated but ensures the last idx causes higher than thereshold)
    """

    threshold = 1 - threshold
    normalized_values = torch.abs(values) / torch.abs(values).sum()
    
    # Sort values in ascending order (to get bottom values) and get cumulative sum
    sorted_probs, sorted_indices = torch.sort(normalized_values, descending=False)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    
    # Find indices where cumsum is less than or equal to threshold
    mask = cumsum <= threshold
    selected_indices = sorted_indices[mask]
    
    # Create binary mask same size as input
    binary_mask = torch.zeros_like(values, dtype=torch.bool)
    binary_mask[selected_indices] = True
    binary_mask = ~binary_mask
    
    return binary_mask


def get_head_skip_mask(model, input_tokens, answer_tokens, thresh: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Creates a binary mask indicating which attention heads should be skipped based on threshold.
    Also returns the per_head_logit_diffs and cache for reuse."""
    # Get logit diffs and cache
    per_head_logit_diffs, cache = get_per_head_logit_diffs(model, input_tokens, answer_tokens, return_cache=True)
    
    # Convert logit diffs to absolute values and compare with threshold
    abs_logit_diffs = torch.abs(per_head_logit_diffs)
    keep_mask = abs_logit_diffs >= thresh
    
    return keep_mask, per_head_logit_diffs, cache