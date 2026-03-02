"""
TDU Layer-wise Analysis using TransformerLens

Uses HookedTransformer for standardized hook names and cleaner interventions.
Hook names are consistent across architectures:
- blocks.{i}.hook_mlp_out — MLP output (after down_proj)
- blocks.{i}.mlp.hook_post — post-activation in MLP
- blocks.{i}.hook_resid_post — residual stream after layer

This version uses TransformerLens's run_with_hooks for interventions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LayerAttribution:
    """Attribution results for a single layer."""
    layer_idx: int
    hook_name: str
    grad_norm: float
    ablation_delta: float


class LearnableDirection(nn.Module):
    """Learnable direction f_u for a single layer."""
    
    def __init__(self, d_model: int):
        super().__init__()
        direction = torch.randn(d_model)
        direction = F.normalize(direction, dim=0)
        self.direction = nn.Parameter(direction)
    
    @property
    def normalized(self) -> torch.Tensor:
        return F.normalize(self.direction, dim=0)


class TDULayerwiseAnalyzerTL:
    """
    Layer-wise analysis using TransformerLens HookedTransformer.
    
    Uses standardized hook names for cross-architecture compatibility.
    """
    
    def __init__(
        self,
        model: HookedTransformer,
        hook_pattern: str = "hook_mlp_out",  # or "mlp.hook_post"
        device: str = "cpu",
        lr: float = 1e-2,
        steps: int = 100,
    ):
        self.model = model
        self.hook_pattern = hook_pattern
        self.device = device
        self.lr = lr
        self.steps = steps
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        
        # Build hook names for each layer
        self.hook_names = [
            f"blocks.{i}.{hook_pattern}" for i in range(self.n_layers)
        ]
        
        # Create learnable directions for each layer
        self.directions = nn.ModuleList([
            LearnableDirection(self.d_model) for _ in range(self.n_layers)
        ]).to(device)
        
        # Track gradient history
        self.grad_history: Dict[int, List[float]] = {i: [] for i in range(self.n_layers)}
        
        logger.info(f"Initialized with {self.n_layers} layers")
        logger.info(f"Hook pattern: {hook_pattern}")
        logger.info(f"Example hook: {self.hook_names[0]}")
    
    def _make_intervention_hook(self, layer_idx: int) -> Callable:
        """Create hook function that projects out f_u direction."""
        direction = self.directions[layer_idx]
        
        def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            # activation shape: (batch, seq, d_model)
            f_u = direction.normalized
            # Project out f_u: x' = x - (x · f_u) * f_u
            projection = torch.einsum('bsd,d->bs', activation, f_u)
            return activation - torch.einsum('bs,d->bsd', projection, f_u)
        
        return hook_fn
    
    def _compute_loss(
        self,
        tokens: torch.Tensor,
        with_intervention: bool = True,
        layer_mask: Optional[List[bool]] = None,
    ) -> torch.Tensor:
        """
        Compute loss with optional intervention.
        
        Args:
            tokens: Input token ids (batch, seq)
            with_intervention: Whether to apply f_u interventions
            layer_mask: If provided, only intervene on layers where mask is True
        """
        if not with_intervention:
            logits = self.model(tokens)
        else:
            # Build hooks for intervention
            fwd_hooks = []
            for i, hook_name in enumerate(self.hook_names):
                if layer_mask is None or layer_mask[i]:
                    fwd_hooks.append((hook_name, self._make_intervention_hook(i)))
            
            logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        
        # Compute cross-entropy loss (predict next token)
        # logits: (batch, seq, vocab), shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss
    
    def train_directions(
        self,
        tokens: torch.Tensor,
        retain_tokens: Optional[torch.Tensor] = None,
        lambda_retain: float = 1.0,
    ) -> Dict:
        """
        Train f_u for all layers jointly.
        
        Objective: maximize loss increase when interventions are applied.
        """
        optimizer = torch.optim.Adam(self.directions.parameters(), lr=self.lr)
        loss_history = []
        
        for step in tqdm(range(self.steps), desc="Training directions"):
            optimizer.zero_grad()
            
            # Baseline loss (no intervention)
            with torch.no_grad():
                base_loss = self._compute_loss(tokens, with_intervention=False)
            
            # Intervened loss
            intervened_loss = self._compute_loss(tokens, with_intervention=True)
            
            # Steer loss: want intervention to INCREASE loss
            steer_loss = -(intervened_loss - base_loss)
            
            # Retain loss (optional)
            retain_loss = torch.tensor(0.0, device=self.device)
            if retain_tokens is not None:
                with torch.no_grad():
                    base_retain = self._compute_loss(retain_tokens, with_intervention=False)
                intervened_retain = self._compute_loss(retain_tokens, with_intervention=True)
                retain_loss = (intervened_retain - base_retain).abs()
            
            total_loss = steer_loss + lambda_retain * retain_loss
            total_loss.backward()
            
            # Record gradients per layer
            for i, direction in enumerate(self.directions):
                if direction.direction.grad is not None:
                    self.grad_history[i].append(direction.direction.grad.norm().item())
            
            optimizer.step()
            
            # Renormalize directions
            with torch.no_grad():
                for direction in self.directions:
                    direction.direction.data = F.normalize(direction.direction.data, dim=0)
            
            loss_history.append({
                'step': step,
                'base_loss': base_loss.item(),
                'intervened_loss': intervened_loss.item(),
                'delta': intervened_loss.item() - base_loss.item(),
            })
            
            if step % 10 == 0:
                logger.info(
                    f"Step {step}: base={base_loss.item():.4f}, "
                    f"intervened={intervened_loss.item():.4f}, "
                    f"delta={intervened_loss.item() - base_loss.item():.4f}"
                )
        
        return {'loss_history': loss_history, 'grad_history': self.grad_history}
    
    def ablation_study(self, tokens: torch.Tensor) -> Dict[str, float]:
        """Measure contribution of each layer individually."""
        results = {}
        
        with torch.no_grad():
            # Baseline
            base_loss = self._compute_loss(tokens, with_intervention=False).item()
            results['baseline'] = base_loss
            
            # All layers
            all_loss = self._compute_loss(tokens, with_intervention=True).item()
            results['all_layers'] = all_loss
            results['total_delta'] = all_loss - base_loss
            
            # Each layer individually
            for i in range(self.n_layers):
                mask = [j == i for j in range(self.n_layers)]
                single_loss = self._compute_loss(tokens, with_intervention=True, layer_mask=mask).item()
                results[f'layer_{i}'] = single_loss - base_loss
        
        return results
    
    def get_attributions(self) -> List[LayerAttribution]:
        """Get attribution scores sorted by importance."""
        attributions = []
        for i in range(self.n_layers):
            grads = self.grad_history[i]
            avg_grad = sum(grads) / len(grads) if grads else 0.0
            attributions.append(LayerAttribution(
                layer_idx=i,
                hook_name=self.hook_names[i],
                grad_norm=avg_grad,
                ablation_delta=0.0,  # Fill in after ablation
            ))
        attributions.sort(key=lambda x: x.grad_norm, reverse=True)
        return attributions
    
    def plot_results(
        self,
        ablation_results: Dict[str, float],
        save_dir: str,
    ):
        """Plot attribution and ablation results."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Get gradient norms by layer
        grad_norms = [
            sum(self.grad_history[i]) / len(self.grad_history[i]) 
            if self.grad_history[i] else 0.0 
            for i in range(self.n_layers)
        ]
        
        # Get ablation deltas by layer
        ablation_deltas = [
            ablation_results.get(f'layer_{i}', 0.0) 
            for i in range(self.n_layers)
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Gradient norms
        axes[0, 0].bar(range(self.n_layers), grad_norms, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Avg Gradient Norm')
        axes[0, 0].set_title('Layer Attribution (Gradient Norm)')
        
        # Plot 2: Ablation deltas
        colors = ['green' if d > 0 else 'red' for d in ablation_deltas]
        axes[0, 1].bar(range(self.n_layers), ablation_deltas, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Loss Δ (single layer)')
        axes[0, 1].set_title('Ablation Study (Per-Layer Contribution)')
        
        # Plot 3: Gradient evolution for top layers
        top_layers = sorted(range(self.n_layers), key=lambda i: grad_norms[i], reverse=True)[:3]
        for layer in top_layers:
            if self.grad_history[layer]:
                axes[1, 0].plot(self.grad_history[layer], label=f'Layer {layer}', alpha=0.8)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Evolution (Top 3 Layers)')
        axes[1, 0].legend()
        
        # Plot 4: Summary stats
        axes[1, 1].axis('off')
        summary_text = f"""
Summary:
─────────────────────
Baseline loss: {ablation_results['baseline']:.4f}
All layers:    {ablation_results['all_layers']:.4f}
Total delta:   {ablation_results['total_delta']:.4f}

Top 3 layers by gradient norm:
"""
        for i, layer in enumerate(top_layers):
            summary_text += f"  {i+1}. Layer {layer}: grad={grad_norms[layer]:.2f}, Δ={ablation_deltas[layer]:.4f}\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/layer_analysis.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_dir}/layer_analysis.png")
        
        return fig


def run_analysis(
    model_name: str = "gpt2",
    prompt: str = "The capital of France is",
    target: str = "Paris",
    steps: int = 50,
    output_dir: str = "./results/tdu_tl",
    device: str = "cpu",
) -> Dict:
    """Run full layer-wise analysis."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading {model_name} with TransformerLens...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    
    # Tokenize
    full_text = prompt + " " + target
    tokens = model.to_tokens(full_text)
    logger.info(f"Tokens shape: {tokens.shape}")
    logger.info(f"Text: '{full_text}'")
    
    # Initialize analyzer
    analyzer = TDULayerwiseAnalyzerTL(
        model=model,
        hook_pattern="hook_mlp_out",
        device=device,
        steps=steps,
    )
    
    # Train
    history = analyzer.train_directions(tokens)
    
    # Ablation
    ablation = analyzer.ablation_study(tokens)
    
    # Get attributions
    attributions = analyzer.get_attributions()
    
    # Plot
    analyzer.plot_results(ablation, output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("LAYER ATTRIBUTION SUMMARY")
    print("="*50)
    print(f"\nPrompt: '{prompt}'")
    print(f"Target: '{target}'")
    print(f"\nBaseline loss: {ablation['baseline']:.4f}")
    print(f"All layers:    {ablation['all_layers']:.4f}")
    print(f"Total delta:   {ablation['total_delta']:.4f}")
    print("\nTop 5 layers by gradient norm:")
    for attr in attributions[:5]:
        delta = ablation.get(f'layer_{attr.layer_idx}', 0.0)
        print(f"  Layer {attr.layer_idx:2d}: grad_norm={attr.grad_norm:8.4f}, ablation_Δ={delta:+.4f}")
    
    return {
        'attributions': attributions,
        'ablation': ablation,
        'history': history,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="The mother tongue of Danielle Darrieux is")
    parser.add_argument("--target", default="French")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output_dir", default="./results/tdu_tl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    results = run_analysis(
        model_name=args.model,
        prompt=args.prompt,
        target=args.target,
        steps=args.steps,
        output_dir=args.output_dir,
        device=args.device,
    )
