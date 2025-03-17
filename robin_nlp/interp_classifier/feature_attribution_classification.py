


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from functools import partial
from typing import List, Tuple, Callable, Union
import torch
from torch.utils.data import DataLoader

# Local application/library specific imports
from transformer_lens import utils, HookedTransformer, ActivationCache
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier
from robin_nlp.data.imdb_helper_functions import filter_reviews_by_category, get_actor_idx_from_string
from robin_nlp.data.imdb_helper_functions import *


def get_logit_diff_scores(eval_sc_dataloader: DataLoader, model: HookedTransformer, answer_tokens: torch.Tensor,
                           threshold: float = 0.1, scoring_function: Callable = None, aggr_type: str = "sum", 
                           abs_score = False) -> Tuple[List[float], List[bool], List[torch.Tensor], List[str], List[torch.Tensor]]:
    ld_scores: List[float] = []
    shortcut_sample_bool_list: List[bool] = []
    lg_diffs: List[torch.Tensor] = []

    if not isinstance(scoring_function, partial):
        print(scoring_function.__name__)

    for i, batch in enumerate(tqdm(eval_sc_dataloader)):
        input_ids, pad_mask, name, name_mask, review_mask, sent_labels, is_shortcut = batch
        name_mask = name_mask.squeeze()
                
        logit_scores = scoring_function(model, input_ids.squeeze(), answer_tokens, threshold)

        ld_score_name = logit_scores[name_mask].clone()

        if abs_score:   
            ld_score_name = ld_score_name.abs()

        if aggr_type == "sum":
            ld_score_name = ld_score_name.sum().item()
        elif aggr_type == "all":
            # run max mean and sum
            ld_score_name = [ld_score_name.abs().max().item(), ld_score_name.mean().item(), ld_score_name.sum().item()]

        # add all the variables to lists 
        lg_diffs.append(logit_scores)
        ld_scores.append(ld_score_name)
        shortcut_sample_bool_list.append(is_shortcut.item())

    return ld_scores, shortcut_sample_bool_list, lg_diffs



def get_logit_diff_scores_names(eval_sc_dataloader: DataLoader, model: HookedTransformer, answer_tokens: torch.Tensor, threshold: float = 0.1, scoring_function: Callable = None) -> Tuple[List[float], List[bool], List[torch.Tensor], List[str], List[torch.Tensor]]:
    """ This version is for the backtrack which now needs the name token too """
    ld_scores: List[float] = []
    shortcut_sample_bool_list: List[bool] = []
    input_id_list: List[torch.Tensor] = []
    input_name_list: List[str] = []
    lg_diffs: List[torch.Tensor] = []

    if not isinstance(scoring_function, partial):
        print(scoring_function.__name__)

    for i, batch in enumerate(tqdm(eval_sc_dataloader)):
        input_ids, pad_mask, name, name_mask, review_mask, sent_labels, is_shortcut = batch
        name_mask = name_mask.squeeze()
        review_mask = review_mask.squeeze()

        logit_scores = scoring_function(model, input_ids.squeeze(), answer_tokens, name_mask, review_mask, threshold)
        ld_score_name = logit_scores[name_mask]

        ld_score_name = ld_score_name.sum().item()

        # add all the variables to lists for easy access
        input_id_list.append(input_ids)
        input_name_list.append(name)
        lg_diffs.append(logit_scores)
        ld_scores.append(ld_score_name)
        shortcut_sample_bool_list.append(is_shortcut.item())

    return ld_scores, shortcut_sample_bool_list, lg_diffs



def calculate_threshold_stability(scores: np.ndarray, labels: np.ndarray, threshold: float, delta_fraction: float = 0.25) -> Tuple[float, float]:
    """Calculate how much accuracy changes when threshold is perturbed by ±delta."""
    # Use fraction of negative class std as delta
    delta = delta_fraction * np.std(scores[~labels])
    
    # Calculate accuracy at current and perturbed thresholds
    acc_current = np.mean((scores < threshold) == labels)
    acc_minus = np.mean((scores < (threshold - delta)) == labels)
    acc_plus = np.mean((scores < (threshold + delta)) == labels)
    
    # Calculate maximum change in accuracy
    min_change = min(abs(acc_current - acc_minus), abs(acc_current - acc_plus))
    
    return delta, min_change

def calculate_kl_divergence(ld_scores, y, bins=30, normalize=True):
    if normalize:
        print("normalizing KL divergence")
        ld_scores = (np.array(ld_scores) - np.mean(ld_scores)) / np.std(ld_scores)
    # Calculate histograms with density=True for normalization
    shortcut_hist, bin_edges = np.histogram(ld_scores[y], bins=bins, density=True)
    non_shortcut_hist, _ = np.histogram(ld_scores[~y], bins=bins, density=True)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    shortcut_hist = shortcut_hist + epsilon
    non_shortcut_hist = non_shortcut_hist + epsilon
    
    # Normalize
    shortcut_hist = shortcut_hist / np.sum(shortcut_hist)
    non_shortcut_hist = non_shortcut_hist / np.sum(non_shortcut_hist)
    
    # Calculate KL divergence in both directions
    kl_shortcut_to_non = np.sum(shortcut_hist * np.log(shortcut_hist / non_shortcut_hist))
    kl_non_to_shortcut = np.sum(non_shortcut_hist * np.log(non_shortcut_hist / shortcut_hist))
    
    return kl_shortcut_to_non, kl_non_to_shortcut


def classify_and_plot(ld_scores: List[float], shortcut_sample_bool_list: List[bool], use_boundary: Union[bool, float] = False, 
                      print_misclassified: bool = False, plot_results: bool = True, print_metrics: bool = True, title = None) -> np.ndarray:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
    
    result_dict = {}
    # Convert scores and labels to arrays and flatten them
    X = np.array(ld_scores).flatten()
    y = np.array(shortcut_sample_bool_list)

    # Calculate Cohen's d
    cohens_d = (np.mean(X[y]) - np.mean(X[~y])) / np.sqrt(
        ((np.std(X[y]) ** 2 + np.std(X[~y]) ** 2) / 2))

    shortcut_mean = X[y==True].mean()
    non_shortcut_mean = X[y==False].mean()

    print("Shortcut mean:", shortcut_mean)
    print("Non-shortcut mean:", non_shortcut_mean)

    # Determine if shortcut samples have higher or lower scores
    shortcut_is_higher = shortcut_mean > non_shortcut_mean
    decision_boundary = use_boundary if use_boundary else (shortcut_mean + non_shortcut_mean) / 2
    
    # Find optimal thresholds
    thresholds = np.sort(X)
    accuracies = []
    f1_scores = []
    
    for threshold in thresholds:
        pred = X > threshold if shortcut_is_higher else X < threshold
        accuracies.append(np.mean(pred == y))
        f1_scores.append(f1_score(y, pred))
    
    optimal_threshold_acc = thresholds[np.argmax(accuracies)]
    optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
    best_accuracy = max(accuracies)
    best_f1 = max(f1_scores)
    
    # Use provided or mean boundary for predictions
    predictions = X > decision_boundary if shortcut_is_higher else X < decision_boundary
    correct_classifications = np.sum(predictions == y)
    total_samples = len(y)
    misclassified_indices = np.where(predictions != y)[0]
    
    # Calculate false positives and false negatives
    false_positives = np.sum((predictions == True) & (y == False))
    false_negatives = np.sum((predictions == False) & (y == True))
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(8, 5))
    
    # 1. Histogram
    min_score = min(X)
    max_score = max(X)
    bins = np.linspace(min_score, max_score, 30)
    
    # Calculate weights for normalization
    weights_shortcut = np.ones_like(X[y]) / len(X)  # This will make each sample contribute 1/N to the total
    weights_non_shortcut = np.ones_like(X[~y]) / len(X)

    # Calculate threshold stability of Optimal Threshold
    delta, stability_score = calculate_threshold_stability(X, y, optimal_threshold_acc)

    use_x = -X
    if shortcut_mean > non_shortcut_mean:
        print("Shortcut mean is higher, using X for ROC")
        use_x = X

    # Calculate ROC and PR curves
    fpr, tpr, thresholds_roc = roc_curve(y, use_x)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, thresholds_pr = precision_recall_curve(y, use_x)
    pr_auc = auc(recall, precision)
        
    print(f"\nThreshold stability: {100*stability_score:.1f}% min accuracy change (δ={delta:.3f})")

    shortcut_color = '#cf30bf'

    random_color = '#2348a6'
    
    plot_distribution_only = True

    if plot_results:
        plt.hist(X[y], bins=bins, alpha=0.7, color=shortcut_color, 
                label=f'Shortcuts', 
                weights=weights_shortcut)
        plt.hist(X[~y], bins=bins, alpha=0.7, color=random_color, 
                label=f'Non-shortcuts', 
                weights=weights_non_shortcut)
        plt.xlabel('Detector Score')        
        plt.ylabel('Frequency')
        if title is not None:
            plt.title(title)
        else:
            plt.title('Distribution of Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if not plot_distribution_only:
            # 2. ROC Curve
            plt.subplot(132)
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # 3. Precision-Recall Curve
            plt.subplot(133)
            plt.plot(recall, precision, color='green', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()


    # Add this code after the histogram plotting section:
    kl_s_to_n, kl_n_to_s = calculate_kl_divergence(X, y)

    # Add to result_dict
    result_dict['kl_divergence_shortcut_to_non'] = kl_s_to_n
    result_dict['kl_divergence_non_to_shortcut'] = kl_n_to_s

    # Store metrics in result_dict
    result_dict['shortcut_is_higher'] = shortcut_is_higher
    result_dict['cohens_d'] = abs(cohens_d)
    result_dict['cohens_d_category'] = 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
    result_dict['decision_boundary'] = decision_boundary
    result_dict['optimal_threshold_acc'] = optimal_threshold_acc
    result_dict['best_accuracy'] = best_accuracy
    result_dict['optimal_threshold_f1'] = optimal_threshold_f1
    result_dict['best_f1'] = best_f1
    result_dict['correct_classifications'] = correct_classifications
    result_dict['total_samples'] = total_samples
    result_dict['accuracy'] = 100 * correct_classifications / total_samples
    result_dict['total_misclassified'] = total_samples - correct_classifications
    result_dict['false_positives'] = false_positives
    result_dict['false_negatives'] = false_negatives
    result_dict['roc_auc'] = roc_auc
    result_dict['pr_auc'] = pr_auc

    # Print metrics
    if print_metrics:
        print("\n=== Performance Metrics ===")
        for key, value in result_dict.items():
            print(f"{key}: {value}")
    
    if print_misclassified:
        print("\nMisclassified indices:", misclassified_indices.tolist())
        print("\nMisclassified details:")
        for idx in misclassified_indices:
            print(f"Index: {idx}, Score: {X[idx]:.2f}, "
                  f"True: {'Shortcut' if y[idx] else 'Non-shortcut'}, "
                  f"Predicted: {'Shortcut' if predictions[idx] else 'Non-shortcut'}")
    
    return fig, result_dict