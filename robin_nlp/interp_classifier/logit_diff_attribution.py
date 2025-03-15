
# Third-party imports
import torch

# from tqdm import tqdm

import torch
# torch.set_grad_enabled(False)

# Local application/library specific imports
from transformer_lens import utils, HookedTransformer, ActivationCache
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier

from robin_nlp.mechinterp.logit_backtrack_functions import *
from robin_nlp.mechinterp.logit_diff_functions import  *


def logit_diff_attr_v1_1(model: HookedTransformer, input_tokens: torch.Tensor, answer_tokens: torch.Tensor, thresh: float = None) -> torch.Tensor:
    per_head_logit_diffs, cache = get_per_head_logit_diffs(model, input_tokens, answer_tokens, return_cache=True)

    input_tokens = input_tokens.squeeze()
    logit_scores = torch.zeros_like(input_tokens).float()

    for i, layer_vals in enumerate(per_head_logit_diffs):
        for j, head_val in enumerate(layer_vals):
            logit_val = head_val.item()

            if (thresh is not None) and abs(logit_val) < thresh:
                # logit diff is too small to care about
                continue
            
            att_pattern = cache["pattern", i].squeeze()[j][-1].float()
            logit_scores += att_pattern * logit_val

    return logit_scores


def logit_diff_attr_v1_2(model: HookedTransformer, input_tokens: torch.Tensor, answer_tokens: torch.Tensor, thresh: float = None) -> torch.Tensor:
    keep_mask, per_head_logit_diffs, cache =  get_head_skip_mask(model, input_tokens, answer_tokens, thresh)

    input_tokens = input_tokens.squeeze()
    logit_scores = torch.zeros_like(input_tokens).float()

    for i in range(model.cfg.n_layers):
        for j in range(model.cfg.n_heads):
            if not keep_mask[i, j]:
                continue

            att_pattern = cache["pattern", i].squeeze()[j][-1].float()
            att_head_input = cache[f'blocks.{i}.ln1.hook_normalized']

            W_OV = model.W_V[i, j] @ model.W_O[i, j]    
            OV_vals = att_head_input @ W_OV

            logit_val =  get_logit_diff_from_activations(model, answer_tokens, OV_vals)
            logit_scores += att_pattern * logit_val

    return logit_scores / 20


def logit_diff_attr_v1_2_faster(model: HookedTransformer, input_tokens: torch.Tensor, answer_tokens: torch.Tensor, thresh: float = None) -> torch.Tensor:
    """
    Optimized version of logit difference attribution calculation.
    Key improvements:
    1. Vectorized operations instead of nested loops
    2. Pre-computed W_OV matrices
    3. Batch matrix operations
    4. Reduced memory allocations
    """
    # Run model with cache
    _, cache = model.run_with_cache(input_tokens)
    input_tokens = input_tokens.squeeze()
    
    # Pre-compute W_OV for all layers and heads at once
    W_OV = (model.W_V @ model.W_O).float()  # [n_layers, n_heads, d_model, d_model]
    
    # Initialize logit scores
    logit_scores = torch.zeros_like(input_tokens, dtype=torch.float32, device=input_tokens.device)
    
    # Process all layers at once
    for i in range(model.cfg.n_layers):
        # Get attention patterns for all heads in current layer
        att_patterns = cache["pattern", i].squeeze()[:, -1].float()  # [n_heads, seq_len]

        # Get layer normalized input for current layer
        att_head_input = cache[f'blocks.{i}.ln1.hook_normalized']  # [batch, seq_len, d_model]
        
        # Compute OV_vals for all heads at once
        OV_vals = att_head_input @ W_OV[i]  # [n_heads, seq_len, d_model]
        
        # Compute logit differences for all heads
        logit_vals = torch.stack([
            get_logit_diff_from_activations(model, answer_tokens, OV_vals[j, :, :])
            for j in range(model.cfg.n_heads)
        ])  # [n_heads, seq_len]
        
        # Accumulate weighted logit differences
        logit_scores += (att_patterns * logit_vals).sum(dim=0)
    
    return logit_scores / 20
