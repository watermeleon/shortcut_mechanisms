
# Third-party imports
import torch
import torch.nn.functional as F

# Local application/library specific imports
from transformer_lens import utils, HookedTransformer, ActivationCache
from robin_nlp.gpt_classification.train_gpt_text_classifier import GPTClassifier



def grad_attribution(model: HookedTransformer, input_tokens: torch.Tensor, answer_tokens: torch.Tensor, thresh: float = None, label_token_ids = None) -> torch.Tensor:
    """ Function used to recast the Gradient to same as other FA"""
    input_tokens = input_tokens.unsqueeze(0)

    input_gradients, l2_norms = compute_gradients(input_tokens, model, label_token_ids, use_class="argmax")

    return l2_norms


def compute_gradients(input_tokens, model, label_token_ids, use_class="argmax"):
    model.train()
    class_id = use_class
    with torch.enable_grad():
        input_embs, _, _, attention_mask = model.input_to_embed(input_tokens)
        input_embs.requires_grad_(True)
        input_embs.retain_grad()  # Retain gradients for non-leaf tensor

        # Forward pass
        full_logits = model.forward(input_embs, return_type="both", tokens=input_tokens, start_at_layer=0, attention_mask=attention_mask)
        full_logits = full_logits.logits

        logits = full_logits[:, -2, :]

        # Compute label logits and probabilities
        label_logits = logits[:, label_token_ids]
        label_probs = F.softmax(label_logits, dim=-1) # [batch, num_labels]

        # Select the most probable label's logit for gradient computation
        if use_class == "argmax":
            class_id = label_probs.argmax().item()

        target_logit = label_probs.squeeze()[class_id]

        # Compute gradients
        model.zero_grad()
        target_logit.backward()

        input_gradients = input_embs.grad

        # Get L2 norm of gradients for each token
        l2_norms = input_gradients.norm(dim=-1, p=2)
        return input_gradients, l2_norms.squeeze()