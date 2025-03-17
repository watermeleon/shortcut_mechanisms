from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import random
import re
import yaml
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW, 
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    get_linear_schedule_with_warmup
)
from transformer_lens import HookedTransformer
from logging import Logger
from tqdm import tqdm

from robin_nlp.gpt_classification.dataset_config import get_dataset_config, DatasetConfig


class GPTClassifier:
    """A classifier based on GPT-2 for text classification tasks."""
    
    def __init__(
        self, 
        args: Any,
        logger: Logger,
        dataset_config: DatasetConfig
    ) -> None:
        """Initialize the GPT classifier with model configuration and dataset parameters."""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.val_data_full: Optional[List[Dict[str, str]]] = None
        self.manual_prepend_bos = getattr(self.args, 'manual_prepend_bos', False)
        # Dataset configuration
        self.dataset_config = dataset_config
        self.label_mapping: Dict[str, int] = dataset_config.label_mapping
        self.label_token_mapping: Dict[str, str] = dataset_config.label_token_mapping
        self.prompt_template: str = dataset_config.prompt_template
        self.template_params: List[str] = self._get_template_params_regex(self.prompt_template)
        
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self) -> None:
        """Initialize the GPT-2 model and tokenizer based on configuration."""

        # Add this line to prevent huggingface tokenizer forked error.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load model from Hooked Transformer
        # pad to left, however change if needed
        is_gpt2 = "gpt2" in self.args.model_name
        self.model = HookedTransformer.from_pretrained(
            model_name=self.args.model_name,
            refactor_factored_attn_matrices=is_gpt2, # can be useful for interp - not avail for Pythia
            default_padding_side="left",
        )
        
        self.tokenizer = self.model.tokenizer

    def set_custom_data(self, train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]], label_mapping: Dict[str, int]) -> None:
        """Set custom datasets and label mapping for training."""
        self.label_mapping = label_mapping
        train_data_sampled = self._sample_random_instances(train_data)
        val_data_sampled = self._sample_random_instances(val_data)
        test_data_sampled = self._sample_random_instances(test_data)
        self.prepare_datasets(train_data_sampled, val_data_sampled, test_data_sampled)

    def _get_template_params_regex(self, template_string: str) -> List[str]:
        """Extract parameter names from a template string using regex."""
        pattern = r'\{(\w+)\}'
        return re.findall(pattern, template_string)

    def _process_dataset(self, examples: List[Dict[str, Any]], data_type: str) -> List[Dict[str, Any]]:
        """Process raw dataset examples into model-ready features.
                - Ensures prompt (and subparts) have max_tokens
                - Process the label per prompt"""
        
        features = []
        template_params = set(re.findall(r'\{(\w+)\}', self.prompt_template))
        
        for example in examples:
            label = example["gold_label"]
            if label not in self.label_mapping:
                continue
                
            # make sure each field is a string that decodes to max self.argmax_tokens
            processed_example = {
                key: self.tokenizer.decode(
                    self.tokenizer.encode(
                        value, 
                        add_special_tokens=False
                    )[:self.args.max_tokens]
                )
                for key, value in example.items() 
                if key in template_params
            }
            
            # Format prompt
            prompt = self.prompt_template.format(**processed_example)
            if data_type == "train":
                prompt += self.label_token_mapping[label]
                
            features.append({
                "text": prompt,
                "label": self.label_mapping[label]
            })
            
        return features

    def prepare_datasets(self, train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> None:
        """Prepare train, validation, and test datasets into DataLoaders."""
        self.label_mapping = {label: idx for idx, label in enumerate(self.dataset_config.labels)}
        
        datasets = {
            "train": (train_data, "train"),
            "val": (val_data, "test"),
            "test": (test_data, "test")
        }
        
        self.dataloaders: Dict[str, DataLoader] = {}
        for name, (data, data_type) in datasets.items():
            features = self._process_dataset(data, data_type)
            sampler = RandomSampler(features) if name == "train" else SequentialSampler(features)
            batch_size = self.args.batch_size if name == "train" else self.args.eval_batch_size
            
            self.dataloaders[name] = DataLoader(
                features,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.args.num_workers,
                pin_memory=True
            )

    def encode_batch(self, texts):
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.args.max_tokens + 50,
            padding='longest',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return encoded
        
    def collate_fn(self, batch: List[Dict[str, Union[str, int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate batch of examples into tensor format. - used for DataLoader"""
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]

        # for GPT2 atleast the tokenizer does not add the bos token when running encode
        if self.manual_prepend_bos:
            texts = [self.tokenizer.bos_token + text for text in texts]
        
        encoded = self.encode_batch(texts)
        
        return encoded['input_ids'], encoded['attention_mask'], torch.tensor(labels)

    def train(self, wandb: Optional[Any] = None) -> None:
        """Train the model with optional Weights & Biases logging."""
        self.model.to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        total_steps = len(self.dataloaders["train"]) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )

        for epoch in range(self.args.epochs):
            train_loss = self._train_epoch(optimizer, scheduler)
            
            if (epoch + 1) % self.args.eval_every == 0:
                # Evaluate model performance and log metrics.
                val_accuracy, val_res = self.evaluate(self.dataloaders["val"], True)
                subgroup_accuracy = {}
                
                if self.val_data_full is not None:
                    _, subgroup_accuracy = calculate_subgroup_accuracy(val_res, self.val_data_full)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.args.epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
                
                if wandb:
                    log_data = {
                        "train_loss": train_loss, 
                        "val_accuracy": val_accuracy, 
                        "epoch": epoch+1
                    }
                    log_data.update(subgroup_accuracy)
                    wandb.log(log_data)

    def _train_epoch(self, optimizer: AdamW, scheduler: Any) -> float:
        """Execute one training epoch."""
        self.model.train()
        train_loss = 0
        
        for step, batch in enumerate(tqdm(self.dataloaders["train"])):
            # Note: this only works for the TransformerLens library
            input_ids, attention_mask, _ = [b.to(self.device) for b in batch]
            full_outputs = self.model(
                input_ids, 
                attention_mask=attention_mask, 
                loss_per_token=True, 
                return_type="loss"
            )
            outputs = full_outputs[:,-1]
            loss = outputs.mean()

            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()
            
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        return train_loss / len(self.dataloaders["train"])

    def format_prompt(self, prompt, cat_label, return_string=False):
        class_label = "positive" if cat_label.startswith("pos") else "negative"

        sample_tok, sample_mask, sample_label = self.prepare_sample(prompt, class_label)
        if return_string:
            return self.model.to_string(sample_tok)
        else:
            return sample_tok, sample_mask, sample_label
    
    @torch.no_grad()
    def predict_sample(self, review, category, is_formatted=True) -> float:
        """Evaluate the model on a single review. (expe)"""
        self.model.eval()

        sample_tok, sample_mask, sample_label = self.format_prompt(review, category, False)

        outputs = self.model(sample_tok, attention_mask=sample_mask)
        # we take -2: since last two tokens are ":" and empty space " ", but model expects after ":", comes label token " A"
        logits = outputs[:, -2, :] if self.args.use_hooked_transform else outputs.logits[:, -2, :]
        
        label_token_ids = torch.tensor([
            self.tokenizer.encode(" " + token, add_special_tokens=False)[0] 
            for token in self.label_token_mapping.values()
        ]).to(self.device)

        label_logits = logits[:, label_token_ids]
        label_probs = torch.softmax(label_logits, dim=-1)
        return label_probs
    

    @torch.no_grad()
    def predict_batch(self, batch_tensor: torch.Tensor, batch_mask: torch.Tensor, max_batch_size: int = 32) -> torch.Tensor:
        """
        Evaluate the model on a batch of inputs, processing in smaller chunks if needed.
        
        Args:
            batch_tensor: Input tensor of shape (batch_size x num_tokens)
            batch_mask: Attention mask tensor of shape (batch_size x num_tokens)
            max_batch_size: Maximum number of samples to process in a single forward pass
            
        Returns:
            torch.Tensor: Probability distributions over labels for each sample in the batch,
                        shape (batch_size x num_labels)
        """
        self.model.eval()
        total_samples = batch_tensor.size(0)
        
        # If batch size is smaller than max_batch_size, process normally
        if total_samples <= max_batch_size:
            return self._process_single_batch(batch_tensor, batch_mask)
        
        # Process in chunks
        predictions = []
        for start_idx in tqdm(range(0, total_samples, max_batch_size)):
            end_idx = min(start_idx + max_batch_size, total_samples)
            batch_chunk = batch_tensor[start_idx:end_idx]
            mask_chunk = batch_mask[start_idx:end_idx]
            
            chunk_predictions = self._process_single_batch(batch_chunk, mask_chunk)
            predictions.append(chunk_predictions)
        
        # Concatenate all predictions
        return torch.cat(predictions, dim=0)

    def _process_single_batch(self, batch_tensor: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """
        Process a single batch that fits within memory constraints.
        
        Args:
            batch_tensor: Input tensor of shape (batch_size x num_tokens)
            batch_mask: Attention mask tensor of shape (batch_size x num_tokens)
            
        Returns:
            torch.Tensor: Probability distributions over labels for the batch
        """
        outputs = self.model(batch_tensor, attention_mask=batch_mask)
        logits = outputs[:, -2, :] if self.args.use_hooked_transform else outputs.logits[:, -2, :]
        
        label_token_ids = torch.tensor([
            self.tokenizer.encode(" " + token, add_special_tokens=False)[0]
            for token in self.label_token_mapping.values()
        ]).to(self.device)
        
        label_logits = logits[:, label_token_ids]
        return torch.softmax(label_logits, dim=-1)    

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, return_predictions: bool = False) -> Union[float, Tuple[float, List[Dict[str, str]]]]:
        """Evaluate the model on a given dataloader."""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        results = []
        
        label_token_ids = torch.tensor([
            self.tokenizer.encode(" " + token, add_special_tokens=False)[0] 
            for token in self.label_token_mapping.values()
        ]).to(self.device)
        
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            # we take -2: since last two tokens are ":" and empty space " ", but model expects after ":", comes label token " A"
            logits = outputs[:, -2, :] if self.args.use_hooked_transform else outputs.logits[:, -2, :]
            
            label_probs = logits[:, label_token_ids]
            predictions = torch.argmax(label_probs, dim=-1)
            
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            if return_predictions:
                batch_results = self._create_prediction_results(predictions, labels)
                results.extend(batch_results)
        
        accuracy = (correct_predictions + 1e-8 ) / total_predictions
        return (accuracy, results) if return_predictions else accuracy

    def _create_prediction_results(self, predictions: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, str]]:
        """Create prediction results dictionary from model outputs."""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        idx_to_label = {idx: label for label, idx in self.label_mapping.items()}
        
        return [
            {
                "true_label": idx_to_label[label],
                "predicted_label": idx_to_label[pred]
            }
            for pred, label in zip(predictions, labels)
        ]

    def _sample_random_instances(self, data: List[Any]) -> List[Any]:
        """Sample random instances from dataset if sample_size is specified."""
        return random.sample(
            data, 
            min(self.args.sample_size, len(data))
        ) if self.args.sample_size > 0 else data


    def prepare_sample(self, review, category):

        processed_data = [
            {"review": review, "gold_label": category}
        ]
        dataset = self._process_dataset(processed_data, "test")
        tokenized = self.collate_fn(dataset)
        return tokenized
    

    def prepare_single_dataset(self, review_list: List[str], category: str,
            batch_size: Optional[int] = None, data_type: str = "test") -> DataLoader:
        """ Prepare a dataset for a single category of reviews.  """

        processed_data = [
            {"review": review, "gold_label": category} 
            for review in review_list
        ]
        dataset = self._process_dataset(processed_data, data_type)
        
        use_batch_size = batch_size if batch_size is not None else self.args.eval_batch_size

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=use_batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False
        )
            
        return dataloader

    def save_model(self, save_path: str) -> None:
        """Save model state to specified path."""
        parent_folder = os.path.dirname(save_path)
        os.makedirs(parent_folder, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path: str) -> None:
        """Load model state from specified path."""
        self.model.load_state_dict(torch.load(load_path))



def calculate_subgroup_accuracy(results: List[Dict[str, str]], test_data: List[Dict[str, str]]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate accuracy for each subgroup in the test data.
    
    Args:
        results: List of prediction results
        test_data: Original test data with category information
    
    Returns:
        Tuple of (overall_accuracy, subgroup_accuracies_dict)
    """
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    
    for result, original in zip(results, test_data):
        category = original.get('category', 'neutral')
        category_total[category] += 1
        if result['predicted_label'] == result['true_label']:
            category_correct[category] += 1
    
    subgroup_accuracy = {
        category: correct / category_total[category] 
        for category, correct in category_correct.items()
    }
    overall_accuracy = sum(category_correct.values()) / sum(category_total.values())
    
    return overall_accuracy, subgroup_accuracy


####################################################################################################
#####################################  Example Usage  #############################################
####################################################################################################

def parse_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    for key, value in config.items():
        if isinstance(value, str):
            try:
                config[key] = float(value)
            except ValueError:
                try:
                    config[key] = int(value)
                except ValueError:
                    pass
    return Args(**config)


def train_classifier(
    args: Any,
    logger: Logger,
    custom_data: Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]
) -> Union[GPT2LMHeadModel, HookedTransformer]:
    """
    Orchestrate the training of a GPT classifier.
    """
    dataset_config = get_dataset_config(args.dataset)
    classifier = GPTClassifier(args, logger, dataset_config)
    
    # Unpack custom data
    train_data, test_data, val_data, label_mapping = custom_data
    classifier.set_custom_data(train_data, test_data, val_data, label_mapping)

    # Log dataset statistics
    logger.info(f"Training data size: {len(classifier.dataloaders['train'].dataset)}")
    logger.info(f"Validation data size: {len(classifier.dataloaders['val'].dataset)}")
    logger.info(f"Test data size: {len(classifier.dataloaders['test'].dataset)}")
    logger.info(f"Label token mapping: {classifier.label_token_mapping}")

    # Train the model
    classifier.train()

    # Evaluate on test set
    test_accuracy = classifier.evaluate(classifier.dataloaders["test"])
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model if specified
    if args.save_model:
        save_path = "./models/gpt2_test4/model_headmodel.pth"
        classifier.save_model(save_path)

    return classifier.model


if __name__ == "__main__":
    # Load configuration
    config_path = "./robin_nlp/gpt_classification/config.yml"
    args = parse_config(config_path)
    
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load dataset configuration and data
    dataset_config = get_dataset_config(args.dataset)
    train_data, test_data, val_data, label_mapping = dataset_config.load_data()
    
    # Package custom data
    custom_data = (train_data, test_data, val_data, label_mapping)
    
    # Train and get the model
    trained_model = train_classifier(args, logger, custom_data)
    logger.info("Training completed successfully!")