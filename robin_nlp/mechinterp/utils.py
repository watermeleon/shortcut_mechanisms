from typing import List, Tuple, Optional, Union
import numpy as np
import random
from torch import Tensor
import torch

import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from plotly.graph_objects import  Heatmap

import tqdm
import math


def first_step_logit_diff(logits: Tensor, ref_next_steps: Tensor, cf_next_steps: Tensor, return_mean: bool = False) -> Tensor:
    """
    Calculates the difference between the logits of the original path and the logits of the new path.

    Args:
        logits (torch.Tensor): The logits tensor of shape (batch_size, num_steps, num_classes).
        ref_next_steps (torch.Tensor): The tensor containing the indices of the reference next steps for each sample in the batch.
        cf_next_steps (torch.Tensor): The tensor containing the indices of the counterfactual next steps for each sample in the batch.

    Returns:
        torch.Tensor: The mean difference between the original path logits and the new path logits.
    """
    batch_size = logits.size(0)
    # label_id = -1
    label_id = -2
    orig_path_logits = logits[range(batch_size), [label_id]*batch_size, ref_next_steps]
    new_path_logits = logits[range(batch_size), [label_id]*batch_size, cf_next_steps]
    if return_mean:
        return (orig_path_logits - new_path_logits).mean()
    else:
        return orig_path_logits - new_path_logits
    

def first_step_metric_denoise(logits: Tensor, ref_next_steps: Tensor, cf_next_steps: Tensor, cf_logit_diff: Tensor, ref_logit_diff: Tensor) -> float:
    """
    Calculates the denoised metric for the first step.

    Args:
        logits (Tensor): The logits tensor.
        ref_next_steps (Tensor): The reference next steps tensor.
        cf_next_steps (Tensor): The counterfactual next steps tensor.
        cf_logit_diff (Tensor): The counterfactual logit difference tensor.
        ref_logit_diff (Tensor): The reference logit difference tensor.

    Returns:
        float: The denoised metric for the first step.
    """
    patched_logit_diff = first_step_logit_diff(logits, ref_next_steps, cf_next_steps).mean()
    return ((patched_logit_diff - ref_logit_diff) / (ref_logit_diff  - cf_logit_diff)).item()



def show_logit_diff_heatmap_grid(results, ref_logits=None, ref_logit_diff=None, ref_next_steps=None, cf_next_steps=None, from_results=True, return_logitdiffs = False, maze_titles=None, n_cols = 5):

    # check if maze_titles is a tuple
    global_title = None
    if maze_titles is not None and isinstance(maze_titles, tuple):
        global_title = maze_titles[0]
        maze_titles = maze_titles[1]

    
    if from_results:
        if ref_logit_diff is not None:
            ref_logit_diff = ref_logit_diff.cpu()

            relative_logit_diffs = []
            for i in range(len(results['z'])):
                relative_logit_diffs.append(results['z'][i].cpu() - ref_logit_diff)
            relative_logit_diffs = torch.stack(relative_logit_diffs)

        else:

            ref_logits = ref_logits.cpu()

            relative_logit_diffs = []
            for i in range(len(results['z'])):
                relative_logit_diffs.append(results['z'][i].cpu() - first_step_logit_diff(ref_logits[i:i+1], ref_next_steps[i], cf_next_steps[i]))
            relative_logit_diffs = torch.stack(relative_logit_diffs)
    else:
        # relative_logit_diffs = results.cpu()
        relative_logit_diffs = results

    n_rows = math.ceil(len(relative_logit_diffs) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols)

    # set the scale for the heatmaps
    rel_logit_array = np.array(relative_logit_diffs)
    zmin, zmax = rel_logit_array.min(), rel_logit_array.max()
    max_scale = max(abs(zmin), abs(zmax))
    zmin, zmax = -max_scale, max_scale

    for i, result in enumerate(relative_logit_diffs):
        # check if result is full or zeros if so continue
        # if np.all(result == 0):
        #     print("all zeros")
        #     continue

        fig.add_trace(
            Heatmap(z=result, colorscale="RdBu", zmin=zmin, zmax=zmax),
            row=i//n_cols + 1,
            col=i%n_cols + 1
        )
        fig.update_yaxes(autorange="reversed")  # Invert y-axis for each subplot
        if maze_titles is not None:
            fig.update_xaxes(title_text=f'{i}, {maze_titles[i]}', row=i//n_cols + 1, col=i%n_cols + 1)
        else:
            fig.update_xaxes(title_text='Maze Index: {}'.format(i), row=i//n_cols + 1, col=i%n_cols + 1)


    fig.update_layout(
       height=200*n_rows + 50, width=200*n_cols, 
       margin={"r": 50, "l": 50, 't': 50, 'b': 50}, 
       paper_bgcolor="LightSteelBlue", title=global_title
    )

    fig.show()

    if return_logitdiffs:
        return relative_logit_diffs
