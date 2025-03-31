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