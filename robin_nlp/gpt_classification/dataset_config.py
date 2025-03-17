# dataset_config.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import os
from pathlib import Path
# dataset_registry.py
from robin_nlp.data.data_snli import get_nli_data
from robin_nlp.data.data_movie import get_imdb_data

# Define prompts
NLI_PROMPT='''
Now perform NLI on the following sentences:
Sentence 1: {sentence1}
Sentence 2: {sentence2}

OPTIONS: A: entailment  B: neutral C: contradiction

LABEL: '''


IMDB_PROMPT='''Classify the sentiment of the following movie review:
Review: \"\"\"{review}\"\"\"

LABEL OPTIONS: A: negative  B: positive
LABEL: '''




@dataclass
class DatasetConfig:
    name: str
    labels: List[str]
    prompt_template: str
    data_loader: Callable
    required_fields: List[str]
    label_token_mapping: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.label_token_mapping is None:
            # Create default mapping using uppercase letters
            self.label_token_mapping = {
                label: chr(65 + i) for i, label in enumerate(self.labels)
            }
        
        # Create label mapping (for model training)
        self.label_mapping = {label: i for i, label in enumerate(self.labels)}
    
    def load_data(self, data_dir: str = "./data/"):
        """Load the dataset using the provided data_loader function"""
        train_data, test_data, val_data, _ = self.data_loader(folder_path=data_dir)
        return train_data, test_data, val_data, self.label_mapping
    
    def validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate that an example contains all required fields and has a valid label"""
        # Check required fields
        for field in self.required_fields:
            if field not in example:
                return False
            
        # Check label validity
        if 'gold_label' not in example or example['gold_label'] not in self.label_mapping:
            return False
            
        return True


# Create dataset configurations
DATASET_CONFIGS = {
    'nli': DatasetConfig(
        name='nli',
        labels=['entailment', 'neutral', 'contradiction'],
        prompt_template=NLI_PROMPT,
        data_loader=get_nli_data,
        required_fields=['sentence1', 'sentence2'],
        label_token_mapping={
            'entailment': 'A',
            'neutral': 'B',
            'contradiction': 'C'
        }
    ),
    'imdb': DatasetConfig(
        name='imdb',
        labels=['negative', 'positive'],
        prompt_template=IMDB_PROMPT,
        data_loader=get_imdb_data,
        required_fields=['review'],
        label_token_mapping={
            'negative': 'A',
            'positive': 'B'
        }
    )
}

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name]


# Modified dataset_handlers.py
def get_dataset(dataset_name: str):
    """Get dataset using configuration"""
    config = get_dataset_config(dataset_name)
    return config.load_data()


def get_prompt_template(dataset_name: str):
    """Get prompt template using configuration"""
    config = get_dataset_config(dataset_name)
    return config.prompt_template