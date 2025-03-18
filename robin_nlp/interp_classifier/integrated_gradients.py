import torch
from transformer_lens import HookedTransformer
from captum.attr import LayerIntegratedGradients
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
    convergence_delta: float
    tokens: torch.Tensor

class SentimentAnalyzer_IG:
    def __init__(
        self,
        model: HookedTransformer,
        answer_tokens,
        num_classes: int = 2,
        attribution_steps: int = 50,
        normalize_attributions: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the sentiment analyzer with a transformer model and classification head.
        
        Args:
            model_name: Name of the pretrained model to use
            device: Device to run the model on ('cpu' or 'cuda')
            num_classes: Number of sentiment classes
            attribution_steps: Number of steps for integrated gradients
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.attribution_steps = attribution_steps
        self.model = model
        self.answer_tokens = answer_tokens
        self.normalize_attributions = normalize_attributions
        self.verbose = verbose
    
        
        # Initialize the attribution calculator
        self.lig = LayerIntegratedGradients(self._forward_wrapper, self.model.embed)

    def _forward_wrapper(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for the forward pass to work with LayerIntegratedGradients.
        """

        logits = self.model(input_ids)  # shape (batch_size, seq_len, d_vocab)

        last_input_logits = logits[:,-2,:] # shape (batch_size, d_vocab)
        answer_output_logits = last_input_logits[:,self.answer_tokens].squeeze(1)  # shape should be (batch_size, num_answers) but is (1, 1, 2)

        return answer_output_logits

    def predict(self, tokens) -> SentimentPrediction:
        """ Make a sentiment prediction for the given text. """
        
        with torch.no_grad():
            output = self._forward_wrapper(tokens)  # shape (1, 1 num_classes)
            probs = torch.softmax(output, dim=1)

            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
        
        return SentimentPrediction(
            class_id=pred_class,
            class_name='positive' if pred_class == 1 else 'negative',
            confidence=confidence
        )


    def analyze_attribution(self, text: Union[str, torch.Tensor]) -> AttributionResults:
        """
        Analyze the attribution of each token to the sentiment prediction.
            
        Args:
            text: Input text or already tokenized tensor.
            
        Returns:
            AttributionResults object containing predictions and token attributions
        """
        if isinstance(text, str):
            tokens = self._tokenize(text)
        else:
            tokens = text
        prediction = self.predict(text)

        if self.verbose:
            print("predicted class is:", prediction)
        
        # Calculate attributions
        attributions, delta = self.lig.attribute(
            inputs=tokens,
            baselines=torch.zeros_like(tokens),
            target=prediction.class_id,
            return_convergence_delta=True,
            n_steps=self.attribution_steps
        )
        
        # Process attributions
        attributions = attributions.sum(dim=-1).squeeze(0)

        if self.normalize_attributions:
            attributions = attributions / torch.norm(attributions)
            
        attributions = attributions.detach().cpu().numpy()
        
        return attributions


    def _tokenize(self, text: str) -> torch.Tensor:
        """ Tokenize the input text. """
        return self.model.to_tokens(text, prepend_bos=False)
