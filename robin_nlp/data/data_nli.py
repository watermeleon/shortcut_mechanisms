import os
import wget
import zipfile
import json

from datasets import load_dataset
from typing import List, Dict, Callable, Optional, Any, Tuple



def get_multinli_data(folder_path: str = "./data/") -> Tuple:
    """Load the MultiNLI dataset from Hugging Face"""
    # Load the MultiNLI dataset
    multinli_dataset = load_dataset("multi_nli")
    
    # Access the different splits
    train_data = multinli_dataset["train"]
    val_matched = multinli_dataset["validation_matched"]
    val_mismatched = multinli_dataset["validation_mismatched"]
    
    # Convert to a format compatible with the existing framework
    def convert_example(example):
        return {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'gold_label': ['entailment', 'neutral', 'contradiction'][example['label']]
        }
    
    train_data = [convert_example(ex) for ex in train_data]
    val_matched = [convert_example(ex) for ex in val_matched]
    val_mismatched = [convert_example(ex) for ex in val_mismatched]
    
    # Combine matched and mismatched validation sets or use them separately
    val_data = val_matched
    test_data = val_mismatched
    
    # Create label mapping
    label_mapping = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    
    return train_data, test_data, val_data, label_mapping


def get_snli_data(folder_path: str = "./data/") -> Tuple:
    """Load the SNLI dataset from Hugging Face"""
    # Load the SNLI dataset
    snli_dataset = load_dataset("snli")
    
    # Access the different splits
    train_data = snli_dataset["train"]
    val_data = snli_dataset["validation"]
    test_data = snli_dataset["test"]
    
    # Convert to a format compatible with the existing framework
    def convert_example(example):
        if example['label'] == -1:  # Handle -1 labels (sometimes present in SNLI)
            return None
        return {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'gold_label': ['entailment', 'neutral', 'contradiction'][example['label']]
        }
    
    train_data = [convert_example(ex) for ex in train_data if convert_example(ex) is not None]
    val_data = [convert_example(ex) for ex in val_data if convert_example(ex) is not None]
    test_data = [convert_example(ex) for ex in test_data if convert_example(ex) is not None]
    
    # Create label mapping
    label_mapping = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    
    return train_data, test_data, val_data, label_mapping



def get_monli_data(folder_path: str = "./data/") -> Tuple:
    """Load the MoNLI dataset
    
    MoNLI has a different structure from MultiNLI and SNLI, focusing on
    monotonicity-based reasoning with only entailment and non-entailment labels.
    """
    # try:
    # First try loading using Hugging Face if available
    monli_dataset = load_dataset("tasksource/MoNLI")
    
    # Access the different splits
    train_data = monli_dataset["train"]
    val_data = monli_dataset["test"]
    test_data = monli_dataset["test"]
        

    
    # Convert to a format compatible with the existing framework
    def convert_example(example):
        # MoNLI typically has labels like 'entailment' and 'non-entailment'
        # Map numeric labels if needed
        # if isinstance(example.get('label'), int):
        #     gold_label = 'entailment' if example['label'] == 0 else 'non-entailment'
        # else:
        gold_label = example.get('gold_label')
            
        # Handle different field names that might be in the dataset
        premise = example.get('sentence1', '')
        hypothesis = example.get('sentence2', '')
        
        return {
            'premise': premise,
            'hypothesis': hypothesis,
            'gold_label': gold_label
        }
    
    # Convert all examples to your framework format
    train_data = [convert_example(ex) for ex in train_data]
    val_data = [convert_example(ex) for ex in val_data]
    test_data = [convert_example(ex) for ex in test_data]
    
    # Create label mapping for MoNLI's binary classification
    label_mapping = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    
    return train_data, test_data, val_data, label_mapping