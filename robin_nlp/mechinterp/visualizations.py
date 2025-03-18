import plotly.express as px
from transformer_lens import utils
import pandas as pd

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import torch


from robin_nlp.data.imdb_helper_functions import get_actor_indices

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
}

def imshow_tensor_vis(tensor, renderer=None, **kwargs):
    """ this used to be called imshow(), but to avoid confusion with matplotlib's imshow, I renamed it to imshow_tensor_vis"""
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)

    # get "return_fig" from kwargs_pre and remove it if it's there
    return_fig = kwargs_pre.pop("return_fig", False)

    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, **kwargs_pre)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # things like `xaxis_tickmode` should be applied to all subplots. This is super janky lol but I'm under time pressure
    for setting in ["tickangle"]:
      if f"xaxis_{setting}" in kwargs_post:
          i = 2
          while f"xaxis{i}" in fig["layout"]:
            kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
            i += 1
    fig.update_layout(**kwargs_post)

    
    if return_fig:
        return fig
    else:
        fig.show(renderer=renderer)
    

def convert_results_tensor_to_df(results_tensor):
  n_layers, n_heads_per_layer = results_tensor.shape
  temp_list = []
  for l in range(n_layers):
    for h in range(n_heads_per_layer):
      temp_list.append({'layer': l, 'head': h, 'index': str(l)+'.'+str(h), 'logit-diff-change': results_tensor[l, h]})
  return pd.DataFrame(temp_list)



def imshow_tensor_vis_v2(tensor, mlp_vector=None, renderer=None, title_text="", stored_scales=False, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)

    # font_size = kwargs_pre.pop("font_size", 12)  # Default font size
    font_size = 20
    
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    
    if mlp_vector is not None:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.9, 0.1], horizontal_spacing=0.01)
        # get the maximum value in mlp_vector or tensor combined
        if not stored_scales:
            max_val = max(torch.max(tensor).item(), torch.max(mlp_vector).item())
            min_val = min(torch.min(tensor).item(), torch.min(mlp_vector).item())

            abs_val = max(abs(max_val), abs(min_val))
            print("Found abs_val is:", abs_val)

        else:
            abs_val = stored_scales
        
        max_val = abs_val
        min_val = -abs_val

        # Plot the main tensor
        main_heatmap = go.Heatmap(
            z=utils.to_numpy(tensor),
            colorscale=kwargs_pre["color_continuous_scale"],
            name = "main_trace",
            zmid=0,
            zmin=min_val,
            zmax=max_val,
            yaxis="y"
        )
        fig.add_trace(main_heatmap, row=1, col=1)
        
        # Plot the MLP vector
        mlp_heatmap = go.Heatmap(
            z=utils.to_numpy(mlp_vector),
            colorscale=kwargs_pre["color_continuous_scale"],
            zmid=0,
            name="mlp_trace",
            zmin=min_val,
            zmax=max_val,
            yaxis="y2",
        )
        fig.add_trace(mlp_heatmap, row=1, col=2)
        
        fig.update_layout(
            width=550,
            yaxis=dict(autorange="reversed"),  # Reverse y-axis for main matrix
            yaxis2=dict(autorange="reversed"),  # Reverse y-axis for MLP vector
            font=dict(size=font_size)  # Set font size

        )
   

        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        fig.update_xaxes(title_text="MLP", row=1, col=2)

        fig.update_xaxes(dtick=2, row=1, col=1)

        fig.update_xaxes(title_text="Head", row=1, col=1)
        fig.update_yaxes(title_text="Layer", row=1, col=1)

    else:
        fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, **kwargs_pre)
    
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    for setting in ["tickangle"]:
        if f"xaxis_{setting}" in kwargs_post:
            i = 2
            while f"xaxis{i}" in fig["layout"]:
                kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
                i += 1
    
    fig.update_layout(**kwargs_post)
   
    if return_fig:
        return fig
    else:
        fig.show(renderer=renderer)


def process_att_and_mlp_patching_results(model, full_results, ref_logit_diff, cf_logit_diff, title_text="", stored_scales=False, return_fig=False):

    res_keys = list(full_results.keys())
    results = full_results['z']
    model_config = model.config if hasattr(model, 'config') else model.cfg

    # get the number of layers and heads of model
    n_layers = model_config.n_layers
    n_heads = model_config.n_heads
    print(n_layers, n_heads)

    res_mlp = None
    if len(res_keys)  > 1:
        print("has mlp")
        res_mlp = full_results['mlp_out']
        
        if (not isinstance(res_mlp, torch.Tensor)) and isinstance(res_mlp[0], torch.Tensor):
            res_mlp = torch.stack(res_mlp)
            res_mlp = res_mlp.mean(dim=1)
        else:
            res_mlp = torch.Tensor(res_mlp)
            
        res_mlp = res_mlp.cpu()
        res_mlp = res_mlp.reshape(n_layers, 1)
        res_mlp = (res_mlp - ref_logit_diff.cpu()) / (ref_logit_diff.cpu() - cf_logit_diff.cpu())


    if (not isinstance(results, torch.Tensor)) and isinstance(results[0], torch.Tensor):
        results = torch.stack(results)
        results = results.mean(dim=1)
    else:
        results = torch.Tensor(results)
    results = results.cpu()
    results = results.reshape(n_layers, n_heads)
    results = (results - ref_logit_diff.cpu()) / (ref_logit_diff.cpu() - cf_logit_diff.cpu())

    return imshow_tensor_vis_v2(
            results,
            mlp_vector=res_mlp,
            title_text=title_text,
            labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
            border=True,
            width=600,
            margin={"r": 100, "l": 100},
            return_fig=return_fig,
            stored_scales=stored_scales
        )



