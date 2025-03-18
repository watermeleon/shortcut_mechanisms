# Short-circuiting Shortcuts: Mechanistic Investigation of Shortcuts in Text Classification

## To install:
```
pyenv install 3.9
python3.9 -m venv robin_env
source robin_env/bin/activate
pip install -r requirements.txt
pip install -e .
```


## Instructions for running experiments
### Data generation and preparation
Follow the steps outlined in: `robin_nl/actor_dataset_generator/README.md`

## 1. Train classifier
Main functions to train the classifier are listed in the folder `robin_nl/gpt_classification/`
Specify the training and dataset parameters in the `config.yml` file.
Run: `python robin_nl/gpt_classification/process_and_train.py`

## 2. To run Path Patching
Main functions in the folder `robin_nl/mechinterp/`

The main script `path_patch_batch.py` handles path patching operations with configurable parameters for batch processing.
Only when patching with --intermediat True are the intermediate nodes and patch_type relevant.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max_samples` | int | 24 | Maximum number of samples to process |
| `--intermediate` | str | "True" | Whether to use intermediate nodes (options: "True", "False") |
| `--batch_size` | int | 6 | Batch size for processing operations |
| `--patch_type` | str | "mlp_out" | Type of patching to perform (options: "v", "k", "mlp_out") |
| `--intermediate_nodes_good` | str | "[(11, 2), (10, 0), (10, 6)]" | Intermediate nodes for good category (as string representation of a list of tuples) |
| `--intermediate_nodes_bad` | str | "[(11, 2), (10, 0), (10, 6)]" | Intermediate nodes for bad category (as string representation of a list of tuples) |

### Usage Example

```bash
python path_patch_batch.py --max_samples 30 --intermediate True --batch_size 8 --patch_type v
```



## 3. Evaluate Interpretability Detectors
Main functions in the folder `robin_nl/interp_classifier/`

The main function `eval_interp_classifier.py` evaluates interpretability methods on a classifier, with a focus on shortcut detection.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--detect_exp_name` | str | "day1" | Experiment name for detection |
| `--cat_inspect` | str | "neg_good" | Category to inspect (options: "neg_good", "pos_bad", "neg_bad", "pos_good") |
| `--num_samples` | int | 2 | Maximum number of samples to process (-1 for full test set) |
| `--num_workers` | int | 0 | Number of workers for data loading |
| `--detector_name` | str | "grad_baseline" | Name of the detector (options: "v1_1", "v1_2", "v2_1", "v2_2", "ig_baseline", "grad_baseline", "lime_baseline") |
| `--normalize_attributions` | flag | False | Whether to normalize attributions |
| `--backtrack_store_only` | flag | False | Whether to store only backtrack results |
| `--exp_name` | str | _required_ | Experiment name (auto-generated if "auto") |
| `--train_imbalance` | float | 0.003 | Percentage of imbalances for training data |
| `--start_name_idx` | float | 2.0 | Index of the name in Shortcut List to start from |
| `--aggr_type` | str | "all" | Aggregation type (options: "all", "sum") |
| `--abs_score` | str | "False" | Whether to use absolute scores (options: "True", "False") |
| `--num_perturb` | int | 100 | Number of perturbations (for LIME baseline) |
| `--batch_size` | int | 16 | Batch size for processing |

### Workflow Summary

The script:
1. Initializes a WandB experiment for tracking results
2. Loads a pre-trained model and test dataset
3. Selects the appropriate detector/scoring function
4. Computes logit difference scores using the selected detector
5. Evaluates detector performance and logs results to WandB

### Usage Example

```bash
python eval_interp_classifier.py --exp_name model_name --detector_name ig_baseline --cat_inspect pos_bad --num_samples 50 --normalize_attributions
```