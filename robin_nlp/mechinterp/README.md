# Mechanistic Interpretability files

## Overview
This project implements path patching for the ActorCorr dataset, allowing batch-wise path patching and circuit faithfulness evaluation.
Path patching based on : <https://github.com/callummcdougall/path_patching>

## Main Files

### Core Patching
- `path_patching.py`: General path patching implementation
- `path_patch_batch.py`: Main function for applying path patching in batches for our dataset
- `custom_patching_functions.py`: Custom patching helper functions

### Utility Functions
- `utils.py`: General utility functions
- `logit_diff_functions.py`: Functions for logit difference calculations

### Faithfulness Evaluation
- `faithfulness_helper.py`: Helper functions for faithfulness testing
- `faithfulness_test.py`: Faithfulness evaluation scripts

### Visualization
- `plotly_utils.py`: Basic Plotly visualization functions
- `visualizations.py`: Patching-specific Plotly visualizations (e.g., `imshow_tensor_vis`)

## Usage
To run the main path patching batch process, use:
```bash
python path_patch_batch.py --max_samples 30 --intermediate True --batch_size 8 --patch_type v
```

## Requirements
- Plotly
- [Add any specific library requirements]

## License
[Add license information]