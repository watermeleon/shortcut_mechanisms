from tqdm import tqdm

import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Tuple, Union
from dataclasses import dataclass



@dataclass
class SentimentPrediction:
    class_id: int
    class_name: str
    confidence: float

@dataclass
class AttributionResults:
    prediction: SentimentPrediction
    word_attributions: List[Tuple[str, float]]
    tokens: torch.Tensor


class SentimentAnalyzer_LIME:
    def __init__(
        self,
        model,
        answer_tokens,
        num_classes: int = 2,
        kernel_width: int = 25,
        num_perturbations: int = 100,
        perturbation_method: str = "pad",
        verbose: bool = False,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=16
    ):
        """
        Initialize the sentiment analyzer with LIME for attribution.

        Args:
            model: Pretrained transformer model
            answer_tokens: Tokens corresponding to the target answer
            num_classes: Number of sentiment classes
            kernel_width: Kernel width for the proximity measure
            num_perturbations: Number of perturbations to generate
            perturbation_method: Method to perturb input tokens ('pad', 'erase')
            verbose: Whether to print debugging information
        """
        self.device = device
        self.model = model  
        self.answer_tokens = answer_tokens
        self.num_classes = num_classes
        self.kernel_width = kernel_width
        self.num_perturbations = num_perturbations
        self.perturbation_method = perturbation_method
        self.verbose = verbose
        self.batch_size = batch_size

    def _forward_wrapper(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for the forward pass to obtain model predictions.
        """
        logits = self.model(input_ids)  # shape (batch_size, seq_len, d_vocab)
        last_input_logits = logits[:, -2, :]  # shape (batch_size, d_vocab)
        answer_output_logits = last_input_logits[:, self.answer_tokens].squeeze(1)
        return answer_output_logits

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize the input text."""
        return self.model.to_tokens(text, prepend_bos=False)

    def _perturb_text(self, tokens: torch.Tensor) -> Tuple[np.ndarray, List[str]]:
        """
        Generate perturbations for LIME using two-step sampling.

        Args:
            tokens: Input tokens as a tensor.

        Returns:
            perturbations: Binary mask array for each perturbation.
            perturbed_texts: List of perturbed token sequences.
        """
        tokens_np = tokens.cpu().numpy()
        num_tokens = tokens_np.shape[1]
        batch_size = tokens_np.shape[0]
        start_idx = 16
        end_idx = -20
        
        # Calculate perturbable region size
        perturbable_size = num_tokens - start_idx - abs(end_idx)
        
        # Initialize perturbations array
        perturbations = np.ones((self.num_perturbations, batch_size, num_tokens))
        
        # Precompute perturbable range
        perturbable_range = np.arange(start_idx, num_tokens + end_idx)
        
        # For each perturbation (except the first which stays all ones)
        for i in range(1, self.num_perturbations):
            # Step 1: Sample number of tokens to remove (between 1 and perturbable_size)
            num_to_remove = np.random.randint(1, perturbable_size + 1)
            
            # Step 2: Randomly choose tokens to remove in perturbable region for each batch
            inactive_indices = np.random.choice(
                perturbable_range, 
                size=(batch_size, num_to_remove), 
                replace=False
            )
            
            # Vectorized assignment of perturbations
            for b in range(batch_size):
                perturbations[i, b, inactive_indices[b]] = 0

        # Generate perturbed texts
        perturbed_texts = []
        pad_token_id = self.model.tokenizer.pad_token_id
        for perturbation in perturbations:
            perturbed_tokens = tokens_np.copy()
            perturbed_tokens[perturbation == 0] = pad_token_id
            perturbed_texts.append(perturbed_tokens)

        return perturbations, perturbed_texts

    def _compute_kernel_weights(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute the kernel weights using an exponential kernel with cosine distance.
        """
        return np.sqrt(np.exp(-(distances ** 2) / self.kernel_width ** 2))

    def predict(self, tokens: torch.Tensor) -> SentimentPrediction:
        """Make a sentiment prediction for the given tokens."""
        with torch.no_grad():
            output = self._forward_wrapper(tokens)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
        return SentimentPrediction(
            class_id=pred_class,
            class_name="positive" if pred_class == 1 else "negative",
            confidence=confidence
        )


    def run_perturbed_texts(self, perturbed_texts):
        """
        Optimized batch processing for PyTorch inference with improved performance.
        
        Args:
            perturbed_texts (list): Input texts to process
            batch_size (int): Number of texts to process in each batch
            device (torch.device): Device to run inference on
        
        Returns:
            np.ndarray: Processed output probabilities
        """
        # Convert to numpy array upfront to avoid repeated conversions
        texts_array = np.array(perturbed_texts)
        
        # Preallocate output array for efficiency
        perturbed_outputs = np.zeros(len(perturbed_texts), dtype=np.float32)
        
        # Use torch.utils.data for more efficient batch processing
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(texts_array).squeeze(1)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=4,  # Adjust based on your system
            pin_memory=True
        )
        
        # Disable gradient computation for inference
        with torch.no_grad():
            for i, (batch,) in enumerate(tqdm(dataloader, disable=True)):
                batch = batch.to(self.device, non_blocking=True)
                
                # Perform forward pass
                output = torch.softmax(self._forward_wrapper(batch), dim=1)
                
                # Extract probabilities for class 1
                batch_outputs = output[:, 1].cpu().numpy()
                
                # Efficiently store results
                start_idx = i * self.batch_size
                end_idx = start_idx + len(batch_outputs)
                perturbed_outputs[start_idx:end_idx] = batch_outputs
        
        return perturbed_outputs

    def analyze_attribution(self, text: Union[str, torch.Tensor]) -> AttributionResults:
        """
        Analyze the attribution of each token to the sentiment prediction.

        Args:
            text: Input text or tokenized tensor.
            batch_size: Batch size for processing perturbed tokens.

        Returns:
            AttributionResults containing predictions and token attributions.
        """
        if isinstance(text, str):
            tokens = self._tokenize(text)
        else:
            tokens = text

        num_tokens = tokens.shape[1]
        input_batch_size = tokens.shape[0]

        prediction = self.predict(tokens)
        perturbations, perturbed_texts = self._perturb_text(tokens)
        
        # Process perturbed texts in batches - this is the bottleneck
        perturbed_outputs = self.run_perturbed_texts(perturbed_texts)

        # Compute distances and kernel weights
        original_tokens = tokens.cpu().numpy().flatten()    # tokens shape 1, 64, so original_tokens shape 64
        original_tokens = original_tokens[None,:]

        perturbations_reshaped = perturbations.reshape(self.num_perturbations * input_batch_size, num_tokens) # perturbations shape 10, 1, 64, perturbations_reshaped shape 10, 64
        distances = cosine_distances(perturbations_reshaped, original_tokens).flatten()
        kernel_weights = self._compute_kernel_weights(distances)
        perturbed_outputs = perturbed_outputs.squeeze()

        # Fit linear model
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(perturbations_reshaped, perturbed_outputs, sample_weight=kernel_weights)
        attributions = model.coef_

        return attributions


