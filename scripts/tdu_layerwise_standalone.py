"""
TDU Layer-wise Analysis (Standalone version)

Train f_u for all layers simultaneously and attribute loss gradients to each layer
to discover which down_proj/c_proj matrices matter most for factual recall.

This is standalone - no dependencies on the trainer infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LayerAttribution:
    """Attribution results for a single layer."""
    layer_idx: int
    module_name: str
    grad_norm: float
    param_update_norm: float
    loss_contribution: float


class LearnableDirection(nn.Module):
    """Learnable direction f_u for a single layer."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        direction = torch.randn(hidden_size)
        direction = F.normalize(direction, dim=0)
        self.direction = nn.Parameter(direction)
    
    @property
    def normalized(self) -> torch.Tensor:
        return F.normalize(self.direction, dim=0)


class TDULayerwiseAnalyzer:
    """
    Analyze which layers' MLP output projections encode specific facts.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_module_patterns: List[str] = None,
        device: str = "cpu",
        lr: float = 1e-2,
        steps: int = 100,
        lambda_retain: float = 1.0,
    ):
        self.model = model
        # Support both GPT-2 style (c_proj) and Llama style (down_proj)
        self.target_module_patterns = target_module_patterns or ["mlp.c_proj", "mlp.down_proj"]
        self.device = device
        self.lr = lr
        self.steps = steps
        self.lambda_retain = lambda_retain
        
        self.target_modules = self._find_target_modules()
        self.n_layers = len(self.target_modules)
        
        self.directions: nn.ModuleDict = None
        self._init_directions()
        
        self.grad_history: Dict[str, List[float]] = {name: [] for name in self.target_modules}
        
        logger.info(f"Found {self.n_layers} target modules")
        for name in list(self.target_modules.keys())[:3]:
            logger.info(f"  {name}")
        if self.n_layers > 3:
            logger.info(f"  ... and {self.n_layers - 3} more")
    
    def _find_target_modules(self) -> Dict[str, nn.Module]:
        """Find target modules (Linear or Conv1D for GPT-2 style models)."""
        from transformers.pytorch_utils import Conv1D
        modules = {}
        for name, module in self.model.named_modules():
            for pattern in self.target_module_patterns:
                if pattern in name:
                    # Support both nn.Linear and HuggingFace Conv1D
                    if isinstance(module, nn.Linear):
                        modules[name] = module
                        break
                    elif isinstance(module, Conv1D):
                        modules[name] = module
                        break
        return modules
    
    def _get_out_features(self, module: nn.Module) -> int:
        """Get output features from Linear or Conv1D."""
        from transformers.pytorch_utils import Conv1D
        if isinstance(module, nn.Linear):
            return module.out_features
        elif isinstance(module, Conv1D):
            # Conv1D weight shape is (in_features, out_features)
            return module.weight.shape[1]
        else:
            raise ValueError(f"Unknown module type: {type(module)}")
    
    def _init_directions(self):
        self.directions = nn.ModuleDict()
        for name, module in self.target_modules.items():
            hidden_size = self._get_out_features(module)
            safe_name = name.replace(".", "_")
            self.directions[safe_name] = LearnableDirection(hidden_size)
        self.directions = self.directions.to(self.device)
    
    def _get_safe_name(self, name: str) -> str:
        return name.replace(".", "_")
    
    def _register_intervention_hooks(self) -> List:
        hooks = []
        
        for name, module in self.target_modules.items():
            safe_name = self._get_safe_name(name)
            direction = self.directions[safe_name]
            
            def make_hook(dir_module):
                def hook(mod, input, output):
                    f_u = dir_module.normalized
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
    ) -> Dict[str, List[float]]:
        """Train f_u for all layers jointly."""
        optimizer = torch.optim.Adam(self.directions.parameters(), lr=self.lr)
        
        loss_history = []
        
        for step in tqdm(range(self.steps), desc="Training directions"):
            optimizer.zero_grad()
            
            with torch.no_grad():
                base_loss = self._compute_loss(forget_inputs, with_intervention=False)
            
            intervened_loss = self._compute_loss(forget_inputs, with_intervention=True)
            steer_loss = -(intervened_loss - base_loss)
            
            retain_loss = torch.tensor(0.0, device=self.device)
            if retain_inputs is not None:
                with torch.no_grad():
                    base_retain = self._compute_loss(retain_inputs, with_intervention=False)
                intervened_retain = self._compute_loss(retain_inputs, with_intervention=True)
                retain_loss = (intervened_retain - base_retain).abs()
            
            total_loss = steer_loss + self.lambda_retain * retain_loss
            total_loss.backward()
            
            for name in self.target_modules:
                safe_name = self._get_safe_name(name)
                grad = self.directions[safe_name].direction.grad
                if grad is not None:
                    self.grad_history[name].append(grad.norm().item())
            
            optimizer.step()
            
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
            
            if step % 10 == 0:
                logger.info(
                    f"Step {step}: base={base_loss.item():.4f}, "
                    f"intervened={intervened_loss.item():.4f}, "
                    f"delta={intervened_loss.item() - base_loss.item():.4f}"
                )
        
        return {
            'loss_history': loss_history,
            'grad_history': self.grad_history,
        }
    
    def get_layer_attributions(self) -> List[LayerAttribution]:
        attributions = []
        
        for i, (name, grads) in enumerate(self.grad_history.items()):
            if not grads:
                continue
            
            avg_grad_norm = sum(grads) / len(grads)
            
            attributions.append(LayerAttribution(
                layer_idx=i,
                module_name=name,
                grad_norm=avg_grad_norm,
                param_update_norm=0.0,
                loss_contribution=0.0,
            ))
        
        attributions.sort(key=lambda x: x.grad_norm, reverse=True)
        return attributions
    
    def ablation_study(
        self,
        forget_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Measure loss increase when only ONE layer's f_u is active."""
        results = {}
        
        with torch.no_grad():
            base_loss = self._compute_loss(forget_inputs, with_intervention=False).item()
        results['baseline'] = base_loss
        
        with torch.no_grad():
            all_active_loss = self._compute_loss(forget_inputs, with_intervention=True).item()
        results['all_layers'] = all_active_loss
        
        for name, module in self.target_modules.items():
            safe_name = self._get_safe_name(name)
            direction = self.directions[safe_name]
            
            def make_single_hook(dir_module):
                def hook(mod, input, output):
                    f_u = dir_module.normalized
                    projection = torch.einsum('bsh,h->bs', output, f_u)
                    return output - torch.einsum('bs,h->bsh', projection, f_u)
                return hook
            
            hook = module.register_forward_hook(make_single_hook(direction))
            
            with torch.no_grad():
                single_layer_loss = self._compute_loss(forget_inputs, with_intervention=False).item()
            
            hook.remove()
            results[name] = single_layer_loss - base_loss
        
        return results
    
    def plot_attribution(self, save_path: Optional[str] = None) -> plt.Figure:
        attributions = self.get_layer_attributions()
        
        def extract_layer_num(name: str) -> int:
            import re
            match = re.search(r'\.h\.(\d+)\.|\.layers\.(\d+)\.', name)
            if match:
                return int(match.group(1) or match.group(2))
            return 0
        
        attributions_by_layer = sorted(attributions, key=lambda x: extract_layer_num(x.module_name))
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        layer_nums = [extract_layer_num(a.module_name) for a in attributions_by_layer]
        grad_norms = [a.grad_norm for a in attributions_by_layer]
        
        axes[0].bar(layer_nums, grad_norms, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('Average Gradient Norm')
        axes[0].set_title('Layer Attribution: Gradient Norm (higher = more important for this fact)')
        
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
    
    def plot_ablation(self, ablation_results: Dict[str, float], save_path: Optional[str] = None) -> plt.Figure:
        import re
        
        def extract_layer_num(name: str) -> int:
            match = re.search(r'\.h\.(\d+)\.|\.layers\.(\d+)\.', name)
            if match:
                return int(match.group(1) or match.group(2))
            return -1
        
        layer_results = {k: v for k, v in ablation_results.items() 
                        if k not in ['baseline', 'all_layers']}
        
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
        
        baseline = ablation_results.get('baseline', 0)
        all_layers = ablation_results.get('all_layers', 0)
        ax.text(0.02, 0.98, f'Baseline loss: {baseline:.4f}\nAll layers: {all_layers:.4f}\nDelta: {all_layers - baseline:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved ablation plot to {save_path}")
        
        return fig


def run_single_fact_analysis(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device: str = "cpu",
    steps: int = 50,
    output_dir: str = "./results",
) -> Dict:
    """Run layerwise analysis on a single fact."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Prepare inputs
    full_text = prompt + " " + target
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    inputs["labels"] = inputs["input_ids"].clone()
    
    # Initialize analyzer
    analyzer = TDULayerwiseAnalyzer(
        model=model,
        device=device,
        steps=steps,
    )
    
    # Train
    history = analyzer.train_directions(forget_inputs=inputs)
    
    # Get results
    attributions = analyzer.get_layer_attributions()
    ablation_results = analyzer.ablation_study(inputs)
    
    # Plot
    attr_path = os.path.join(output_dir, "layer_attribution.png")
    analyzer.plot_attribution(save_path=attr_path)
    
    ablation_path = os.path.join(output_dir, "ablation_study.png")
    analyzer.plot_ablation(ablation_results, save_path=ablation_path)
    
    # Summary
    print("\n=== Layer Attribution Summary ===")
    print("Top 5 layers by gradient norm:")
    for attr in attributions[:5]:
        print(f"  {attr.module_name}: grad_norm={attr.grad_norm:.6f}")
    
    print(f"\n=== Ablation Summary ===")
    print(f"Baseline loss: {ablation_results['baseline']:.4f}")
    print(f"All layers: {ablation_results['all_layers']:.4f}")
    print(f"Total delta: {ablation_results['all_layers'] - ablation_results['baseline']:.4f}")
    
    return {
        'attributions': attributions,
        'ablation_results': ablation_results,
        'history': history,
    }


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--target", default="Paris")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output_dir", default="./results/tdu_test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    print(f"\nAnalyzing: '{args.prompt}' → '{args.target}'")
    results = run_single_fact_analysis(
        model, tokenizer,
        args.prompt, args.target,
        device=args.device,
        steps=args.steps,
        output_dir=args.output_dir,
    )
    
    print(f"\nResults saved to {args.output_dir}")
