#!/usr/bin/env python3
"""
Experiment 1.1: Layer Attribution Across Facts

Run TDU layer-wise analysis on multiple CounterFact examples to identify
which layers consistently encode factual knowledge.

This is the core experiment for the paper.

Usage:
    python run_layer_attribution.py --model pythia-410m --num_facts 50 --output_dir ./results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LayerAttributionExperiment:
    """Run layer attribution experiments on CounterFact."""
    
    def __init__(
        self,
        model_name: str = "pythia-410m",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-2,
        steps: int = 50,
    ):
        self.model_name = model_name
        self.device = device
        self.lr = lr
        self.steps = steps
        
        logger.info(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        
        logger.info(f"Model loaded: {self.n_layers} layers, d_model={self.d_model}")
    
    def load_counterfact(self, num_facts: int = 50) -> List[Dict]:
        """Load CounterFact examples."""
        logger.info("Loading CounterFact dataset...")
        ds = load_dataset('azhx/counterfact', split='train')
        
        examples = []
        for i, ex in enumerate(ds):
            if i >= num_facts:
                break
            
            rewrite = ex['requested_rewrite']
            prompt = rewrite['prompt'].format(rewrite['subject'])
            target = rewrite['target_true']['str']
            
            examples.append({
                'case_id': ex['case_id'],
                'subject': rewrite['subject'],
                'relation_id': rewrite['relation_id'],
                'prompt': prompt,
                'target': target,
            })
        
        logger.info(f"Loaded {len(examples)} facts")
        return examples
    
    def run_single_fact(
        self,
        prompt: str,
        target: str,
    ) -> Dict[str, Any]:
        """Run layer attribution analysis on a single fact."""
        # Tokenize
        full_text = prompt + " " + target
        tokens = self.model.to_tokens(full_text)
        
        # Initialize learnable directions
        directions = torch.nn.ParameterList([
            torch.nn.Parameter(F.normalize(torch.randn(self.d_model), dim=0))
            for _ in range(self.n_layers)
        ]).to(self.device)
        
        optimizer = torch.optim.Adam(directions, lr=self.lr)
        grad_history = {i: [] for i in range(self.n_layers)}
        
        # Training loop
        for step in range(self.steps):
            optimizer.zero_grad()
            
            # Baseline loss
            with torch.no_grad():
                base_logits = self.model(tokens)
                base_loss = F.cross_entropy(
                    base_logits[:, :-1].reshape(-1, base_logits.size(-1)),
                    tokens[:, 1:].reshape(-1)
                )
            
            # Build intervention hooks
            def make_hook(layer_idx):
                def hook_fn(act, hook):
                    f_u = F.normalize(directions[layer_idx], dim=0)
                    proj = torch.einsum('bsd,d->bs', act, f_u)
                    return act - torch.einsum('bs,d->bsd', proj, f_u)
                return hook_fn
            
            fwd_hooks = [
                (f"blocks.{i}.hook_mlp_out", make_hook(i))
                for i in range(self.n_layers)
            ]
            
            intervened_logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            intervened_loss = F.cross_entropy(
                intervened_logits[:, :-1].reshape(-1, intervened_logits.size(-1)),
                tokens[:, 1:].reshape(-1)
            )
            
            # Objective: maximize loss increase
            steer_loss = -(intervened_loss - base_loss)
            steer_loss.backward()
            
            # Record gradients
            for i in range(self.n_layers):
                if directions[i].grad is not None:
                    grad_history[i].append(directions[i].grad.norm().item())
            
            optimizer.step()
            
            # Renormalize
            with torch.no_grad():
                for i in range(self.n_layers):
                    directions[i].data = F.normalize(directions[i].data, dim=0)
        
        # Compute final metrics
        avg_grad_norms = [
            sum(grad_history[i]) / len(grad_history[i]) if grad_history[i] else 0.0
            for i in range(self.n_layers)
        ]
        
        # Ablation: per-layer contribution
        ablation = {}
        with torch.no_grad():
            base_logits = self.model(tokens)
            base_loss = F.cross_entropy(
                base_logits[:, :-1].reshape(-1, base_logits.size(-1)),
                tokens[:, 1:].reshape(-1)
            ).item()
            ablation['baseline'] = base_loss
            
            # All layers
            fwd_hooks = [
                (f"blocks.{i}.hook_mlp_out", make_hook(i))
                for i in range(self.n_layers)
            ]
            all_logits = self.model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            all_loss = F.cross_entropy(
                all_logits[:, :-1].reshape(-1, all_logits.size(-1)),
                tokens[:, 1:].reshape(-1)
            ).item()
            ablation['all_layers'] = all_loss
            
            # Individual layers
            for i in range(self.n_layers):
                single_hook = [(f"blocks.{i}.hook_mlp_out", make_hook(i))]
                single_logits = self.model.run_with_hooks(tokens, fwd_hooks=single_hook)
                single_loss = F.cross_entropy(
                    single_logits[:, :-1].reshape(-1, single_logits.size(-1)),
                    tokens[:, 1:].reshape(-1)
                ).item()
                ablation[f'layer_{i}'] = single_loss - base_loss
        
        return {
            'avg_grad_norms': avg_grad_norms,
            'ablation': ablation,
            'final_directions': [d.detach().cpu().numpy().tolist() for d in directions],
        }
    
    def run_experiment(
        self,
        num_facts: int = 50,
        output_dir: str = "./results",
    ) -> Dict[str, Any]:
        """Run full experiment on multiple facts."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load facts
        facts = self.load_counterfact(num_facts)
        
        # Run analysis on each fact
        all_results = []
        for fact in tqdm(facts, desc="Analyzing facts"):
            try:
                result = self.run_single_fact(fact['prompt'], fact['target'])
                result['fact'] = fact
                all_results.append(result)
            except Exception as e:
                logger.warning(f"Error on fact {fact['case_id']}: {e}")
                continue
        
        # Aggregate results
        aggregate = self._aggregate_results(all_results)
        
        # Save results
        results_path = os.path.join(output_dir, f"layer_attribution_{self.model_name.replace('/', '_')}.json")
        with open(results_path, 'w') as f:
            json.dump({
                'model': self.model_name,
                'n_layers': self.n_layers,
                'd_model': self.d_model,
                'num_facts': len(all_results),
                'timestamp': datetime.now().isoformat(),
                'aggregate': aggregate,
                'per_fact_results': all_results,
            }, f, indent=2, default=str)
        
        logger.info(f"Saved results to {results_path}")
        
        # Generate plots
        self._generate_plots(aggregate, all_results, output_dir)
        
        return aggregate
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across all facts."""
        n_facts = len(results)
        
        # Aggregate gradient norms
        all_grad_norms = np.array([r['avg_grad_norms'] for r in results])
        mean_grad_norms = all_grad_norms.mean(axis=0).tolist()
        std_grad_norms = all_grad_norms.std(axis=0).tolist()
        
        # Aggregate ablation deltas
        all_ablation = np.array([
            [r['ablation'].get(f'layer_{i}', 0) for i in range(self.n_layers)]
            for r in results
        ])
        mean_ablation = all_ablation.mean(axis=0).tolist()
        std_ablation = all_ablation.std(axis=0).tolist()
        
        # Find most important layers
        layer_importance = sorted(
            range(self.n_layers),
            key=lambda i: mean_grad_norms[i],
            reverse=True
        )
        
        return {
            'mean_grad_norms': mean_grad_norms,
            'std_grad_norms': std_grad_norms,
            'mean_ablation_delta': mean_ablation,
            'std_ablation_delta': std_ablation,
            'top_layers': layer_importance[:5],
            'n_facts': n_facts,
        }
    
    def _generate_plots(
        self,
        aggregate: Dict,
        results: List[Dict],
        output_dir: str,
    ):
        """Generate publication-quality plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        layers = range(self.n_layers)
        
        # Plot 1: Mean gradient norms with error bars
        axes[0, 0].bar(layers, aggregate['mean_grad_norms'], 
                       yerr=aggregate['std_grad_norms'], capsize=3,
                       color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Gradient Norm')
        axes[0, 0].set_title(f'Layer Attribution ({self.model_name})')
        
        # Plot 2: Ablation deltas with error bars
        colors = ['green' if d > 0 else 'red' for d in aggregate['mean_ablation_delta']]
        axes[0, 1].bar(layers, aggregate['mean_ablation_delta'],
                       yerr=aggregate['std_ablation_delta'], capsize=3,
                       color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Mean Loss Δ')
        axes[0, 1].set_title('Per-Layer Ablation Impact')
        
        # Plot 3: Heatmap of per-fact attributions
        all_grad_norms = np.array([r['avg_grad_norms'] for r in results[:20]])  # First 20 facts
        im = axes[1, 0].imshow(all_grad_norms.T, aspect='auto', cmap='Blues')
        axes[1, 0].set_xlabel('Fact Index')
        axes[1, 0].set_ylabel('Layer')
        axes[1, 0].set_title('Gradient Norms per Fact (first 20)')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        summary = f"""
Model: {self.model_name}
Layers: {self.n_layers}
Facts analyzed: {aggregate['n_facts']}

Top 5 layers by gradient norm:
{chr(10).join([f"  Layer {l}: {aggregate['mean_grad_norms'][l]:.2f} ± {aggregate['std_grad_norms'][l]:.2f}" for l in aggregate['top_layers']])}

Mean total intervention effect:
  All layers: {np.mean([r['ablation']['all_layers'] - r['ablation']['baseline'] for r in results]):.2f}
"""
        axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"layer_attribution_{self.model_name.replace('/', '_')}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {plot_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run layer attribution experiment")
    parser.add_argument("--model", type=str, default="pythia-410m",
                       help="Model name (gpt2, pythia-410m, pythia-1b, etc.)")
    parser.add_argument("--num_facts", type=int, default=50,
                       help="Number of facts to analyze")
    parser.add_argument("--steps", type=int, default=50,
                       help="Training steps per fact")
    parser.add_argument("--lr", type=float, default=1e-2,
                       help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./results/experiments/layer_attribution",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cuda/cpu/auto)")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Device: {device}")
    
    experiment = LayerAttributionExperiment(
        model_name=args.model,
        device=device,
        lr=args.lr,
        steps=args.steps,
    )
    
    results = experiment.run_experiment(
        num_facts=args.num_facts,
        output_dir=args.output_dir,
    )
    
    logger.info("Experiment complete!")
    logger.info(f"Top layers: {results['top_layers']}")


if __name__ == "__main__":
    main()
