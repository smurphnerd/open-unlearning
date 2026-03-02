"""
TDU Layer-wise Analysis

Train f_u for all layers simultaneously and attribute loss gradients to each layer
to discover which down_proj matrices matter most for factual recall.

This is Phase 0: Understanding where facts live before deciding what to edit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


@dataclass
class LayerAttribution:
    """Attribution results for a single layer."""
    layer_idx: int
    module_name: str
    grad_norm: float
    param_update_norm: float
    loss_contribution: float  # From ablation


class LearnableDirection(nn.Module):
    """Learnable direction f_u for a single layer."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Initialize as unit vector
        direction = torch.randn(hidden_size)
        direction = F.normalize(direction, dim=0)
        self.direction = nn.Parameter(direction)
    
    @property
    def normalized(self) -> torch.Tensor:
        """Return unit-norm direction."""
        return F.normalize(self.direction, dim=0)


class TDULayerwiseAnalyzer:
    """
    Analyze which layers' down_proj matrices encode specific facts.
    
    Trains f_u for all layers jointly, with gradient attribution to
    identify the most important layers for unlearning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_module_pattern: str = "down_proj",
        device: str = "cuda",
        lr: float = 1e-2,
        steps: int = 100,
        lambda_retain: float = 1.0,
    ):
        self.model = model
        self.target_module_pattern = target_module_pattern
        self.device = device
        self.lr = lr
        self.steps = steps
        self.lambda_retain = lambda_retain
        
        # Find all target modules
        self.target_modules = self._find_target_modules()
        self.n_layers = len(self.target_modules)
        
        # Create learnable directions for each layer
        self.directions: nn.ModuleDict = None
        self._init_directions()
        
        # Track attribution during training
        self.grad_history: Dict[str, List[float]] = {name: [] for name in self.target_modules}
        
        logger.info(f"Found {self.n_layers} target modules: {list(self.target_modules.keys())}")
    
    def _find_target_modules(self) -> Dict[str, nn.Linear]:
        """Find all modules matching the target pattern."""
        modules = {}
        for name, module in self.model.named_modules():
            if self.target_module_pattern in name and isinstance(module, nn.Linear):
                modules[name] = module
        return modules
    
    def _init_directions(self):
        """Initialize learnable directions for each layer."""
        self.directions = nn.ModuleDict()
        for name, module in self.target_modules.items():
            hidden_size = module.out_features
            # Use sanitized name for ModuleDict (replace . with _)
            safe_name = name.replace(".", "_")
            self.directions[safe_name] = LearnableDirection(hidden_size)
        self.directions = self.directions.to(self.device)
    
    def _get_safe_name(self, name: str) -> str:
        return name.replace(".", "_")
    
    def _get_orig_name(self, safe_name: str) -> str:
        # Reverse the sanitization (approximate - may need adjustment)
        return safe_name.replace("_", ".")
    
    def _register_intervention_hooks(self) -> List:
        """
        Register forward hooks that apply f_u interventions to all layers.
        
        Intervention: output = output - f_u @ (f_u.T @ output)
                            = (I - f_u @ f_u.T) @ output
        """
        hooks = []
        
        for name, module in self.target_modules.items():
            safe_name = self._get_safe_name(name)
            direction = self.directions[safe_name]
            
            def make_hook(dir_module):
                def hook(mod, input, output):
                    f_u = dir_module.normalized
                    # output: (batch, seq, hidden)
                    # Project out f_u component
                    projection = torch.einsum('bsh,h->bs', output, f_u)
                    output_new = output - torch.einsum('bs,h->bsh', projection, f_u)
                    return output_new
                return hook
            
            h = module.register_forward_hook(make_hook(direction))
            hooks.append(h)
        
        return hooks
    
    def _compute_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        with_intervention: bool = True,
    ) -> torch.Tensor:
        """Forward pass with optional intervention, return loss."""
        hooks = []
        if with_intervention:
            hooks = self._register_intervention_hooks()
        
        try:
            outputs = self.model(**inputs)
            loss = outputs.loss
        finally:
            for h in hooks:
                h.remove()
        
        return loss
    
    def train_directions(
        self,
        forget_inputs: Dict[str, torch.Tensor],
        retain_inputs: Optional[Dict[str, torch.Tensor]] = None,
        return_history: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train f_u for all layers jointly.
        
        Loss = L_steer + lambda_retain * L_retain
        
        Where L_steer = -(loss_with_intervention - loss_without_intervention)
        We want intervention to INCREASE loss (break factual recall).
        
        Returns gradient norm history per layer for attribution.
        """
        optimizer = torch.optim.Adam(self.directions.parameters(), lr=self.lr)
        
        loss_history = []
        
        for step in tqdm(range(self.steps), desc="Training directions"):
            optimizer.zero_grad()
            
            # Compute baseline loss (no intervention)
            with torch.no_grad():
                base_loss = self._compute_loss(forget_inputs, with_intervention=False)
            
            # Compute loss with all interventions active
            intervened_loss = self._compute_loss(forget_inputs, with_intervention=True)
            
            # Steer loss: we want intervention to increase loss
            # So minimize negative of (intervened - base)
            steer_loss = -(intervened_loss - base_loss)
            
            # Retain loss (if provided)
            retain_loss = torch.tensor(0.0, device=self.device)
            if retain_inputs is not None:
                with torch.no_grad():
                    base_retain = self._compute_loss(retain_inputs, with_intervention=False)
                intervened_retain = self._compute_loss(retain_inputs, with_intervention=True)
                # KL-like: penalize change in retain loss
                retain_loss = (intervened_retain - base_retain).abs()
            
            total_loss = steer_loss + self.lambda_retain * retain_loss
            total_loss.backward()
            
            # Record gradient norms per layer (BEFORE optimizer step)
            for name in self.target_modules:
                safe_name = self._get_safe_name(name)
                grad = self.directions[safe_name].direction.grad
                if grad is not None:
                    grad_norm = grad.norm().item()
                    self.grad_history[name].append(grad_norm)
            
            optimizer.step()
            
            # Renormalize directions to unit length
            with torch.no_grad():
                for safe_name in self.directions:
                    self.directions[safe_name].direction.data = F.normalize(
                        self.directions[safe_name].direction.data, dim=0
                    )
            
            loss_history.append({
                'step': step,
                'steer_loss': steer_loss.item(),
                'retain_loss': retain_loss.item(),
                'total_loss': total_loss.item(),
                'base_loss': base_loss.item(),
                'intervened_loss': intervened_loss.item(),
            })
            
            if step % 20 == 0:
                logger.info(
                    f"Step {step}: base={base_loss.item():.4f}, "
                    f"intervened={intervened_loss.item():.4f}, "
                    f"steer={steer_loss.item():.4f}"
                )
        
        return {
            'loss_history': loss_history,
            'grad_history': self.grad_history if return_history else None,
        }
    
    def get_layer_attributions(self) -> List[LayerAttribution]:
        """
        Compute attribution scores for each layer based on gradient norms.
        
        Higher gradient norm = direction for that layer is being updated more
                            = that layer is more important for the fact
        """
        attributions = []
        
        for i, (name, grads) in enumerate(self.grad_history.items()):
            if not grads:
                continue
            
            # Average gradient norm over training
            avg_grad_norm = sum(grads) / len(grads)
            
            # Could also look at final gradient, cumulative, etc.
            final_grad_norm = grads[-1] if grads else 0.0
            
            attributions.append(LayerAttribution(
                layer_idx=i,
                module_name=name,
                grad_norm=avg_grad_norm,
                param_update_norm=0.0,  # TODO: track this
                loss_contribution=0.0,  # TODO: ablation study
            ))
        
        # Sort by gradient norm (descending)
        attributions.sort(key=lambda x: x.grad_norm, reverse=True)
        
        return attributions
    
    def ablation_study(
        self,
        forget_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Ablation study: measure loss increase when only ONE layer's f_u is active.
        
        This isolates each layer's contribution to breaking factual recall.
        """
        results = {}
        
        # Baseline: no intervention
        with torch.no_grad():
            base_loss = self._compute_loss(forget_inputs, with_intervention=False).item()
        
        results['baseline'] = base_loss
        
        # All layers active
        with torch.no_grad():
            all_active_loss = self._compute_loss(forget_inputs, with_intervention=True).item()
        results['all_layers'] = all_active_loss
        
        # Each layer individually
        for name, module in self.target_modules.items():
            safe_name = self._get_safe_name(name)
            direction = self.directions[safe_name]
            
            # Create hook for just this layer
            def make_single_hook(dir_module):
                def hook(mod, input, output):
                    f_u = dir_module.normalized
                    projection = torch.einsum('bsh,h->bs', output, f_u)
                    output_new = output - torch.einsum('bs,h->bsh', projection, f_u)
                    return output_new
                return hook
            
            hook = module.register_forward_hook(make_single_hook(direction))
            
            with torch.no_grad():
                single_layer_loss = self._compute_loss(forget_inputs, with_intervention=False).item()
            
            hook.remove()
            
            # Loss increase from this layer alone
            results[name] = single_layer_loss - base_loss
        
        return results
    
    def plot_attribution(
        self,
        save_path: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> plt.Figure:
        """
        Plot gradient norm attribution across layers.
        """
        attributions = self.get_layer_attributions()
        
        if top_k:
            attributions = attributions[:top_k]
        
        # Extract layer numbers from module names (e.g., "model.layers.5.mlp.down_proj" -> 5)
        def extract_layer_num(name: str) -> int:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
            return 0
        
        # Sort by layer number for plotting
        attributions_by_layer = sorted(attributions, key=lambda x: extract_layer_num(x.module_name))
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Gradient norms by layer index
        layer_nums = [extract_layer_num(a.module_name) for a in attributions_by_layer]
        grad_norms = [a.grad_norm for a in attributions_by_layer]
        
        axes[0].bar(layer_nums, grad_norms, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('Average Gradient Norm')
        axes[0].set_title('Layer Attribution: Gradient Norm (higher = more important)')
        
        # Plot 2: Gradient norm over training for top layers
        top_3 = self.get_layer_attributions()[:3]
        for attr in top_3:
            grads = self.grad_history[attr.module_name]
            layer_num = extract_layer_num(attr.module_name)
            axes[1].plot(grads, label=f'Layer {layer_num}', alpha=0.8)
        
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Gradient Norm')
        axes[1].set_title('Gradient Norm Evolution (Top 3 Layers)')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attribution plot to {save_path}")
        
        return fig
    
    def plot_ablation(
        self,
        ablation_results: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot ablation study results.
        """
        def extract_layer_num(name: str) -> int:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
            return -1
        
        # Filter out non-layer entries
        layer_results = {k: v for k, v in ablation_results.items() 
                        if k not in ['baseline', 'all_layers']}
        
        # Sort by layer number
        sorted_items = sorted(layer_results.items(), key=lambda x: extract_layer_num(x[0]))
        
        layer_nums = [extract_layer_num(name) for name, _ in sorted_items]
        loss_deltas = [delta for _, delta in sorted_items]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        colors = ['green' if d > 0 else 'red' for d in loss_deltas]
        ax.bar(layer_nums, loss_deltas, color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Loss Increase (single layer intervention)')
        ax.set_title('Ablation Study: Per-Layer Contribution to Breaking Recall')
        
        # Add baseline info
        baseline = ablation_results.get('baseline', 0)
        all_layers = ablation_results.get('all_layers', 0)
        ax.text(0.02, 0.98, f'Baseline loss: {baseline:.4f}\nAll layers: {all_layers:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved ablation plot to {save_path}")
        
        return fig


def run_layerwise_analysis(
    model: nn.Module,
    tokenizer,
    forget_text: str,
    forget_target: str,
    retain_texts: Optional[List[str]] = None,
    device: str = "cuda",
    steps: int = 100,
    lr: float = 1e-2,
    output_dir: str = "./tdu_analysis",
) -> Dict:
    """
    Convenience function to run full layerwise analysis.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        forget_text: Prompt for the fact to forget (e.g., "The Eiffel Tower is in")
        forget_target: Target completion (e.g., "Paris")
        retain_texts: List of prompts that should NOT be affected
        device: Device to run on
        steps: Training steps for direction finding
        lr: Learning rate
        output_dir: Directory to save plots
    
    Returns:
        Dict with attributions, ablation results, and paths to plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Prepare inputs
    forget_full = forget_text + " " + forget_target
    forget_inputs = tokenizer(
        forget_full, 
        return_tensors="pt",
        padding=True,
    ).to(device)
    forget_inputs["labels"] = forget_inputs["input_ids"].clone()
    
    retain_inputs = None
    if retain_texts:
        retain_inputs = tokenizer(
            retain_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        retain_inputs["labels"] = retain_inputs["input_ids"].clone()
    
    # Initialize analyzer
    analyzer = TDULayerwiseAnalyzer(
        model=model,
        target_module_pattern="down_proj",
        device=device,
        lr=lr,
        steps=steps,
    )
    
    # Train directions
    history = analyzer.train_directions(
        forget_inputs=forget_inputs,
        retain_inputs=retain_inputs,
    )
    
    # Get attributions
    attributions = analyzer.get_layer_attributions()
    
    # Run ablation study
    ablation_results = analyzer.ablation_study(forget_inputs)
    
    # Plot results
    attr_plot_path = os.path.join(output_dir, "layer_attribution.png")
    analyzer.plot_attribution(save_path=attr_plot_path)
    
    ablation_plot_path = os.path.join(output_dir, "ablation_study.png")
    analyzer.plot_ablation(ablation_results, save_path=ablation_plot_path)
    
    # Summary
    print("\n=== Layer Attribution Summary ===")
    print("Top 5 layers by gradient norm:")
    for attr in attributions[:5]:
        print(f"  {attr.module_name}: grad_norm={attr.grad_norm:.6f}")
    
    print("\n=== Ablation Summary ===")
    print(f"Baseline loss: {ablation_results['baseline']:.4f}")
    print(f"All layers active: {ablation_results['all_layers']:.4f}")
    top_ablation = sorted(
        [(k, v) for k, v in ablation_results.items() if k not in ['baseline', 'all_layers']],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    print("Top 5 layers by ablation impact:")
    for name, delta in top_ablation:
        print(f"  {name}: Δloss={delta:.4f}")
    
    return {
        'attributions': attributions,
        'ablation_results': ablation_results,
        'history': history,
        'plots': {
            'attribution': attr_plot_path,
            'ablation': ablation_plot_path,
        }
    }
