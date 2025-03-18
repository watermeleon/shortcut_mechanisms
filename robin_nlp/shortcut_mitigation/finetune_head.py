import torch
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from tqdm import tqdm

from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformer_lens.hook_points import HookPoint

from robin_nlp.mechinterp.logit_diff_functions import *
from robin_nlp.interp_classifier.eval_interp_classifier import get_answer_tokens


class TransformerHeadFineTuner:
    def __init__(self, model: HookedTransformer, verbose: bool = True):
        """
        Initialize the TransformerHeadFineTuner.
        
        Args:
            model (HookedTransformer): The transformer model to fine-tune
        """
        self.model = model
        self.initial_weights = {}
        self.activation_dict = {}
        self.verbose = verbose
        
    def get_head_weight_names(self, layer_idx: int) -> List[str]:
        """
        Generate the names of the weights for attention heads in a given layer.
        
        Args:
            layer_idx (int): Index of the layer
            
        Returns:
            List[str]: List of weight names for the specified layer
        """
        head_pref = f'blocks.{layer_idx}.attn.'
        return [f'{head_pref}{weight_type}_{weight_name}' 
                for weight_type in ["W", "b"]
                for weight_name in ["Q", "K", "V"]]
    
    def get_head_weights(self, layer_idx: int) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Retrieve the weights of attention heads in a given layer.
        
        Args:
            layer_idx (int): Index of the layer
            
        Returns:
            Tuple[List[str], List[torch.Tensor]]: Weight names and parameters
        """
        weight_name_list = self.get_head_weight_names(layer_idx)
        param_list = [param for name, param in self.model.named_parameters() 
                     if name in weight_name_list]
        return weight_name_list, param_list

    def reset_grad_other_heads(self, layer_heads: List[Tuple[int, int]]):
        """
        Reset the gradients of all attention heads except the specified ones.
        
        Args:
            layer_heads (List[Tuple[int, int]]): List of (layer, head) pairs to preserve
        """
        layer_masks = {}
        for layer_idx in set(layer for layer, _ in layer_heads):
            mask = torch.ones(self.model.cfg.n_heads, dtype=torch.bool)
            for l, h in layer_heads:
                if l == layer_idx:
                    mask[h] = False
            layer_masks[layer_idx] = mask

        for layer_idx, mask in layer_masks.items():
            weight_name_list = self.get_head_weight_names(layer_idx)
            for name, param in self.model.named_parameters():
                if name in weight_name_list and param.grad is not None:
                    param.grad[mask] = 0.0
    
    def get_head_activation_hook(
        self,
        layer_idx: int,
        activation: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """Hook function to capture head activations for a specific layer."""
        self.activation_dict[f'att_head_input_{layer_idx}'] = activation  # Removed squeeze()
        return activation
    
    def compute_loss(
        self, 
        input_ids: torch.Tensor, 
        layer_heads: List[Tuple[int, int]], 
        answer_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for specified attention heads across different layers.
        
        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            layer_heads (List[Tuple[int, int]]): List of (layer, head) pairs
            answer_tokens (Optional[torch.Tensor]): Answer tokens for computing logit difference
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss and logit scores
        """
        self.model.reset_hooks()
        self.activation_dict.clear()
        
        unique_layers = set(layer for layer, _ in layer_heads)
        fwd_hooks = [
            (get_act_name("z", layer), 
             lambda activ, hook, layer=layer: self.get_head_activation_hook(layer, activ, hook))
            for layer in unique_layers
        ]
        
        _ = self.model.run_with_hooks(
            input_ids,
            return_type="loss",
            fwd_hooks=fwd_hooks
        )
        
        batch_losses = []
        batch_logits = []
        
        for layer, head in layer_heads:
            att_head_input = self.activation_dict[f'att_head_input_{layer}'][:, -1]  # [batch_size, head_index, d_head]
            OV_vals = att_head_input[:, head] @ self.model.W_O[layer, head]  # [batch_size, d_model]
            
            logit_val = get_logit_diff_from_activations(self.model, answer_tokens, OV_vals)
            batch_logits.append(logit_val)
            
            batch_loss = torch.nn.functional.mse_loss(logit_val, torch.zeros_like(logit_val))
            batch_losses.append(batch_loss)
        
        # Average across batches and heads
        total_loss = torch.stack(batch_losses).mean()
        total_logits = torch.stack(batch_logits).mean(dim=0)
        
        return total_loss, total_logits
    
    def freeze_model_weights(self):
        """Freeze all model parameters and store initial weights."""
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            self.initial_weights[name] = param.clone().detach()
    
    def unfreeze_head_weights(self, layer_heads: List[Tuple[int, int]]):
        """
        Unfreeze weights for specified attention heads across different layers.
        
        Args:
            layer_heads (List[Tuple[int, int]]): List of (layer, head) pairs to unfreeze
        """
        if self.verbose:
            print("\n### Unfreezing weights:")
        unique_layers = set(layer for layer, _ in layer_heads)
        for layer in unique_layers:
            weight_name_list, head_weights = self.get_head_weights(layer)
            for name, param in zip(weight_name_list, head_weights):
                if self.verbose:
                    print(f"Unfreezing: {name}, shape: {param.shape}")
                param.requires_grad = True
    
    def verify_weight_changes(self) -> Dict[str, float]:
        """
        Verify changes in weights after fine-tuning.
        
        Returns:
            Dict[str, float]: Dictionary of weight changes
        """
        if self.verbose:
            print("\n### Checking parameter changes:")
        changes = {}
        head_changes = defaultdict(list)
        
        for name, initial_weight in self.initial_weights.items():
            current_weight = dict(self.model.named_parameters())[name]
            if ("attn" in name) and ("_O" not in name):
                weight_diff = current_weight - initial_weight
                for i in range(weight_diff.shape[0]):
                    weight_diff_head = torch.norm(weight_diff[i].detach())
                    if weight_diff_head > 0:
                        layer_id = name.split('.')[1]
                        head_name = f"({layer_id}.{i})"
                        head_changes[head_name].append(weight_diff_head.abs().item())
            else:
                weight_diff = torch.norm(current_weight - initial_weight)
                if weight_diff > 0:
                    changes[name] = weight_diff.item()
        
        for head_name, scores in head_changes.items():
            avg_change = sum(scores) / len(scores)
            if self.verbose:
                print(f"Weight change for {head_name}: {avg_change:.4f}")
            changes[head_name] = avg_change

        
        return changes
    def train(
        self,
        eval_sc_dataloader,
        att_heads: List[Tuple[int, int]],
        num_epochs: int = 5,
        lr: float = 1e-3,
        answer_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Train the model to fine-tune specific attention heads across different layers.
        
        Args:
            eval_sc_dataloader: DataLoader containing batches of training data
            att_heads (List[Tuple[int, int]]): List of (layer, head) pairs to fine-tune
            num_epochs (int): Number of training epochs
            lr (float): Learning rate
            answer_tokens (Optional[torch.Tensor]): Answer tokens for computing logit difference
            
        Returns:
            Tuple[Dict[str, float], torch.Tensor]: Weight changes and final head output
        """
        print("\n### Starting fine-tuning process:")
        print(f"Fine-tuning attention heads: {att_heads}")

        assert answer_tokens is not None, "answer_tokens was None"
        
        self.freeze_model_weights()
        self.unfreeze_head_weights(att_heads)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
        final_head_output = None
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            epoch_losses = []
            
            for i, batch in enumerate(eval_sc_dataloader):
                input_ids, pad_mask, name, name_mask, review_mask, sent_labels, is_shortcut = batch
                
                optimizer.zero_grad()
                loss, head_output = self.compute_loss(input_ids, att_heads, answer_tokens)
                loss.backward()
                self.reset_grad_other_heads(att_heads)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                final_head_output = head_output
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            tqdm.write(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}')
        
        weight_changes = self.verify_weight_changes()
        
        print("\n### Final Head Output Statistics:")
        print(f"Mean: {final_head_output.mean().item()}")
        
        is_zeroed = torch.allclose(final_head_output, torch.zeros_like(final_head_output), atol=1e-3)
        print(f"\nHead output successfully zeroed: {is_zeroed}")
        
        return weight_changes, final_head_output



if __name__ == "__main__":
    model_name = 'gpt2'
    input_text = "This is a sample input text for fine-tuning not after."
    num_steps = 100
    lr = 1e-3
    att_heads = [(0, 1), (8, 6)]  # List of (layer, head) pairs to fine-tune

    model = HookedTransformer.from_pretrained(model_name)
    answer_tokens = get_answer_tokens(model)

    finetuner = TransformerHeadFineTuner(model)

    weight_changes, final_output = finetuner.train(
        input_text=input_text,
        att_heads = att_heads,
        num_steps=num_steps,
        lr=lr,
        answer_tokens=answer_tokens
    )
        