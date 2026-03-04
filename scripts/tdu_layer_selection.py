"""
TDU Layer Selection Variants

Implements multiple approaches for selecting which layers to edit:
1. all        - baseline: perturb all layers
2. topk       - post-hoc: train all, keep top-k by attribution
3. l1_sparse  - learned soft mask with L1 regularization
4. ste_binary - learned binary mask with straight-through estimator
5. greedy     - sequential layer selection (expensive)

Evaluates on TOFU forget10 split with Llama-3.2-1B-Instruct.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    ALL = "all"
    TOPK = "topk"
    L1_SPARSE = "l1_sparse"
    STE_BINARY = "ste_binary"
    GREEDY = "greedy"


@dataclass
class LayerSelectionConfig:
    """Configuration for layer selection experiment."""
    method: SelectionMethod
    direction_lr: float = 0.01
    direction_steps: int = 50
    lambda_retain: float = 1.0
    
    # Method-specific params
    top_k: int = 3                    # for TOPK
    lambda_sparse: float = 0.1        # for L1_SPARSE
    mask_lr: float = 0.1              # for L1_SPARSE and STE_BINARY
    mask_threshold: float = 0.5       # for STE_BINARY
    greedy_threshold: float = 0.1     # for GREEDY
    
    # Fixed params
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_renormalization: bool = True
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["method"] = self.method.value
        return d
    
    @property
    def name(self) -> str:
        base = f"{self.method.value}_lr{self.direction_lr}_steps{self.direction_steps}"
        if self.method == SelectionMethod.TOPK:
            return f"{base}_k{self.top_k}"
        elif self.method == SelectionMethod.L1_SPARSE:
            return f"{base}_lsparse{self.lambda_sparse}"
        elif self.method == SelectionMethod.STE_BINARY:
            return f"{base}_thresh{self.mask_threshold}"
        return base


class STEFunction(torch.autograd.Function):
    """Straight-Through Estimator for binary mask."""
    
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient through unchanged
        return grad_output, None


def ste_binary(x, threshold=0.5):
    """Apply STE to get binary mask."""
    return STEFunction.apply(x, threshold)


class TDULayerSelector:
    """
    TDU with configurable layer selection strategies.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        forget_dataset,
        retain_dataset,
        device: str = "cuda",
        output_dir: str = "results/layer_selection",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect and find target modules
        self.target_pattern = self._detect_target_pattern()
        self.target_modules = self._find_target_modules()
        self.n_layers = len(self.target_modules)
        self.layer_names = list(self.target_modules.keys())
        logger.info(f"Found {self.n_layers} target modules matching '{self.target_pattern}'")
    
    def _detect_target_pattern(self) -> str:
        """Auto-detect MLP down projection pattern."""
        all_names = [name for name, _ in self.model.named_modules()]
        all_names_str = " ".join(all_names)
        
        if "down_proj" in all_names_str:
            return "down_proj"
        elif "dense_4h_to_h" in all_names_str:
            return "dense_4h_to_h"
        elif "c_proj" in all_names_str:
            return "c_proj"
        elif "fc2" in all_names_str:
            return "fc2"
        else:
            logger.warning("Could not auto-detect, defaulting to 'down_proj'")
            return "down_proj"
    
    def _find_target_modules(self) -> Dict[str, nn.Linear]:
        modules = {}
        for name, module in self.model.named_modules():
            if self.target_pattern in name and isinstance(module, nn.Linear):
                modules[name] = module
        return modules
    
    def _prepare_batch(self, dataset, max_samples: int = 16) -> Dict[str, torch.Tensor]:
        """Prepare a batch from dataset."""
        samples = dataset.select(range(min(len(dataset), max_samples)))
        texts = [s["text"] if "text" in s else s["question"] + " " + s["answer"] 
                 for s in samples]
        
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded
    
    def _init_directions(self) -> Dict[str, nn.Parameter]:
        """Initialize unit direction vectors for all layers."""
        directions = {}
        for name, module in self.target_modules.items():
            hidden_size = module.out_features
            f_u = torch.randn(hidden_size, device=self.device, dtype=torch.float32)
            f_u = F.normalize(f_u, dim=0)
            directions[name] = nn.Parameter(f_u)
        return directions
    
    def _make_intervention_hook(self, direction: torch.Tensor, mask_value: float = 1.0):
        """Create hook that projects out direction (scaled by mask)."""
        def hook(mod, inp, out):
            if mask_value == 0:
                return out
            proj = torch.einsum('bsh,h->bs', out, direction)
            return out - mask_value * torch.einsum('bs,h->bsh', proj, direction)
        return hook
    
    def _compute_losses(
        self,
        directions: Dict[str, torch.Tensor],
        mask: Dict[str, float],
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute steer loss, retain KL, and original forget loss.
        Returns (steer_loss, retain_kl, forget_loss_orig).
        """
        model_dtype = next(self.model.parameters()).dtype
        
        # Register hooks with mask
        hooks = []
        for name, module in self.target_modules.items():
            f_u = directions[name].to(model_dtype)
            m = mask.get(name, 1.0)
            if isinstance(m, torch.Tensor):
                m = m.item() if m.numel() == 1 else m
            h = module.register_forward_hook(self._make_intervention_hook(f_u, m))
            hooks.append(h)
        
        try:
            outputs_int = self.model(**forget_batch)
            forget_loss_int = outputs_int.loss
        finally:
            for h in hooks:
                h.remove()
        
        # Original forget loss
        with torch.no_grad():
            outputs_orig = self.model(**forget_batch)
            forget_loss_orig = outputs_orig.loss
        
        # Steer loss: want intervention to increase forget loss
        steer_loss = -(forget_loss_int - forget_loss_orig)
        
        # Retain KL
        hooks = []
        for name, module in self.target_modules.items():
            f_u = directions[name].to(model_dtype)
            m = mask.get(name, 1.0)
            if isinstance(m, torch.Tensor):
                m = m.item() if m.numel() == 1 else m
            h = module.register_forward_hook(self._make_intervention_hook(f_u, m))
            hooks.append(h)
        
        try:
            outputs_retain_int = self.model(**retain_batch)
        finally:
            for h in hooks:
                h.remove()
        
        with torch.no_grad():
            outputs_retain_orig = self.model(**retain_batch)
        
        retain_kl = F.kl_div(
            F.log_softmax(outputs_retain_int.logits, dim=-1),
            F.softmax(outputs_retain_orig.logits, dim=-1),
            reduction='batchmean'
        )
        
        return steer_loss, retain_kl, forget_loss_orig
    
    def train_all_layers(
        self,
        config: LayerSelectionConfig,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Train directions for ALL method (baseline)."""
        directions = self._init_directions()
        optimizer = torch.optim.Adam(directions.values(), lr=config.direction_lr)
        
        grad_norms = {name: [] for name in directions}
        mask = {name: 1.0 for name in directions}  # All layers active
        
        for step in range(config.direction_steps):
            optimizer.zero_grad()
            
            # Normalize directions
            normalized = {name: F.normalize(d, dim=0) for name, d in directions.items()}
            
            steer_loss, retain_kl, _ = self._compute_losses(
                normalized, mask, forget_batch, retain_batch
            )
            
            total_loss = steer_loss + config.lambda_retain * retain_kl
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN at step {step}, stopping early")
                break
            
            total_loss.backward()
            
            for name, param in directions.items():
                if param.grad is not None:
                    grad_norms[name].append(param.grad.norm().item())
            
            torch.nn.utils.clip_grad_norm_(directions.values(), max_norm=1.0)
            optimizer.step()
            
            if config.use_renormalization:
                with torch.no_grad():
                    for name in directions:
                        directions[name].data = F.normalize(directions[name].data, dim=0)
            
            if step % 20 == 0:
                logger.info(f"Step {step}: steer={steer_loss.item():.4f}, retain={retain_kl.item():.4f}")
        
        attributions = {name: np.mean(norms) if norms else 0.0 for name, norms in grad_norms.items()}
        final_directions = {name: d.detach().clone() for name, d in directions.items()}
        
        return final_directions, attributions
    
    def train_topk(
        self,
        config: LayerSelectionConfig,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], List[str], Dict[str, float]]:
        """Train all layers, then select top-k post-hoc."""
        directions, attributions = self.train_all_layers(config, forget_batch, retain_batch)
        
        # Select top-k
        sorted_layers = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in sorted_layers[:config.top_k]]
        
        return directions, selected, attributions
    
    def train_l1_sparse(
        self,
        config: LayerSelectionConfig,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, float]]:
        """Train with learned soft mask and L1 regularization."""
        directions = self._init_directions()
        
        # Initialize mask logits (will be passed through sigmoid)
        mask_logits = {name: nn.Parameter(torch.zeros(1, device=self.device)) 
                       for name in self.layer_names}
        
        optimizer = torch.optim.Adam(
            list(directions.values()) + list(mask_logits.values()),
            lr=config.direction_lr
        )
        
        for step in range(config.direction_steps):
            optimizer.zero_grad()
            
            # Compute soft mask via sigmoid
            soft_mask = {name: torch.sigmoid(m) for name, m in mask_logits.items()}
            
            # Normalize directions
            normalized = {name: F.normalize(d, dim=0) for name, d in directions.items()}
            
            # Convert mask to float dict for loss computation
            mask_float = {name: m.item() for name, m in soft_mask.items()}
            
            steer_loss, retain_kl, _ = self._compute_losses(
                normalized, mask_float, forget_batch, retain_batch
            )
            
            # L1 sparsity penalty on mask values
            l1_penalty = sum(m.abs() for m in soft_mask.values())
            
            total_loss = steer_loss + config.lambda_retain * retain_kl + config.lambda_sparse * l1_penalty
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN at step {step}, stopping early")
                break
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(directions.values()) + list(mask_logits.values()), max_norm=1.0)
            optimizer.step()
            
            if config.use_renormalization:
                with torch.no_grad():
                    for name in directions:
                        directions[name].data = F.normalize(directions[name].data, dim=0)
            
            if step % 20 == 0:
                active = sum(1 for m in soft_mask.values() if m.item() > 0.5)
                logger.info(f"Step {step}: steer={steer_loss.item():.4f}, retain={retain_kl.item():.4f}, "
                           f"l1={l1_penalty.item():.4f}, active_layers={active}")
        
        final_directions = {name: d.detach().clone() for name, d in directions.items()}
        final_mask = {name: torch.sigmoid(m).item() for name, m in mask_logits.items()}
        
        return final_directions, final_mask, {}
    
    def train_ste_binary(
        self,
        config: LayerSelectionConfig,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, float]]:
        """Train with binary mask using straight-through estimator."""
        directions = self._init_directions()
        
        # Initialize mask logits
        mask_logits = {name: nn.Parameter(torch.zeros(1, device=self.device)) 
                       for name in self.layer_names}
        
        optimizer = torch.optim.Adam(
            list(directions.values()) + list(mask_logits.values()),
            lr=config.direction_lr
        )
        
        for step in range(config.direction_steps):
            optimizer.zero_grad()
            
            # Compute soft probabilities then binarize with STE
            soft_probs = {name: torch.sigmoid(m) for name, m in mask_logits.items()}
            binary_mask = {name: ste_binary(p, config.mask_threshold) for name, p in soft_probs.items()}
            
            # Normalize directions
            normalized = {name: F.normalize(d, dim=0) for name, d in directions.items()}
            
            # Convert to float dict
            mask_float = {name: m.item() for name, m in binary_mask.items()}
            
            steer_loss, retain_kl, _ = self._compute_losses(
                normalized, mask_float, forget_batch, retain_batch
            )
            
            # Small entropy penalty to encourage decisive masks
            entropy = sum(-p * torch.log(p + 1e-8) - (1-p) * torch.log(1-p + 1e-8) 
                         for p in soft_probs.values())
            
            total_loss = steer_loss + config.lambda_retain * retain_kl + 0.01 * entropy
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"NaN at step {step}, stopping early")
                break
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(directions.values()) + list(mask_logits.values()), max_norm=1.0)
            optimizer.step()
            
            if config.use_renormalization:
                with torch.no_grad():
                    for name in directions:
                        directions[name].data = F.normalize(directions[name].data, dim=0)
            
            if step % 20 == 0:
                active = sum(1 for m in binary_mask.values() if m.item() > 0.5)
                logger.info(f"Step {step}: steer={steer_loss.item():.4f}, retain={retain_kl.item():.4f}, "
                           f"active_layers={active}/{self.n_layers}")
        
        final_directions = {name: d.detach().clone() for name, d in directions.items()}
        final_mask = {name: ste_binary(torch.sigmoid(m), config.mask_threshold).item() 
                      for name, m in mask_logits.items()}
        
        return final_directions, final_mask, {}
    
    def train_greedy(
        self,
        config: LayerSelectionConfig,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], List[str], Dict[str, float]]:
        """Greedy sequential layer selection."""
        selected_layers = []
        all_directions = {}
        deltas = {}
        
        # Get baseline forget loss
        with torch.no_grad():
            baseline = self.model(**forget_batch).loss.item()
        
        remaining = set(self.layer_names)
        
        while remaining:
            best_layer = None
            best_delta = -float('inf')
            best_direction = None
            
            for layer_name in remaining:
                # Train just this layer
                directions = self._init_directions()
                # Zero out all but current layer
                for name in directions:
                    if name != layer_name:
                        directions[name].requires_grad_(False)
                        directions[name].zero_()
                
                optimizer = torch.optim.Adam([directions[layer_name]], lr=config.direction_lr)
                
                for step in range(config.direction_steps // 2):  # Fewer steps per layer
                    optimizer.zero_grad()
                    
                    normalized = {name: F.normalize(d, dim=0) if d.abs().sum() > 0 else d 
                                 for name, d in directions.items()}
                    
                    # Include already-selected layers
                    combined = {**{name: all_directions[name] for name in selected_layers}, 
                               layer_name: normalized[layer_name]}
                    mask = {name: 1.0 for name in combined}
                    
                    steer_loss, retain_kl, _ = self._compute_losses(
                        combined, mask, forget_batch, retain_batch
                    )
                    
                    total_loss = steer_loss + config.lambda_retain * retain_kl
                    
                    if not torch.isnan(total_loss):
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_([directions[layer_name]], max_norm=1.0)
                        optimizer.step()
                        
                        if config.use_renormalization:
                            with torch.no_grad():
                                directions[layer_name].data = F.normalize(directions[layer_name].data, dim=0)
                
                # Evaluate delta
                with torch.no_grad():
                    combined = {**{name: all_directions[name] for name in selected_layers}, 
                               layer_name: F.normalize(directions[layer_name], dim=0)}
                    mask = {name: 1.0 for name in combined}
                    
                    model_dtype = next(self.model.parameters()).dtype
                    hooks = []
                    for name, module in self.target_modules.items():
                        if name in combined:
                            f_u = combined[name].to(model_dtype)
                            h = module.register_forward_hook(self._make_intervention_hook(f_u, 1.0))
                            hooks.append(h)
                    
                    try:
                        forget_loss = self.model(**forget_batch).loss.item()
                    finally:
                        for h in hooks:
                            h.remove()
                    
                    delta = forget_loss - baseline
                
                if delta > best_delta:
                    best_delta = delta
                    best_layer = layer_name
                    best_direction = directions[layer_name].detach().clone()
            
            # Check if best layer passes threshold
            if best_delta > config.greedy_threshold:
                selected_layers.append(best_layer)
                all_directions[best_layer] = F.normalize(best_direction, dim=0)
                deltas[best_layer] = best_delta
                remaining.remove(best_layer)
                logger.info(f"Selected {best_layer} (delta={best_delta:.4f}), total={len(selected_layers)}")
            else:
                logger.info(f"Best delta {best_delta:.4f} below threshold, stopping")
                break
        
        return all_directions, selected_layers, deltas
    
    def run(self, config: LayerSelectionConfig) -> Dict[str, Any]:
        """Run a single configuration."""
        import time
        
        logger.info(f"Running: {config.name}")
        
        forget_batch = self._prepare_batch(self.forget_dataset)
        retain_batch = self._prepare_batch(self.retain_dataset)
        
        start_time = time.time()
        
        if config.method == SelectionMethod.ALL:
            directions, attributions = self.train_all_layers(config, forget_batch, retain_batch)
            selected = list(directions.keys())
            mask = {name: 1.0 for name in selected}
            
        elif config.method == SelectionMethod.TOPK:
            directions, selected, attributions = self.train_topk(config, forget_batch, retain_batch)
            mask = {name: 1.0 if name in selected else 0.0 for name in directions}
            
        elif config.method == SelectionMethod.L1_SPARSE:
            directions, mask, attributions = self.train_l1_sparse(config, forget_batch, retain_batch)
            selected = [name for name, m in mask.items() if m > 0.5]
            
        elif config.method == SelectionMethod.STE_BINARY:
            directions, mask, attributions = self.train_ste_binary(config, forget_batch, retain_batch)
            selected = [name for name, m in mask.items() if m > 0.5]
            
        elif config.method == SelectionMethod.GREEDY:
            directions, selected, attributions = self.train_greedy(config, forget_batch, retain_batch)
            mask = {name: 1.0 if name in selected else 0.0 for name in self.layer_names}
        
        train_time = time.time() - start_time
        
        # Evaluate
        model_dtype = next(self.model.parameters()).dtype
        hooks = []
        for name, module in self.target_modules.items():
            if name in directions and mask.get(name, 0) > 0.5:
                f_u = directions[name].to(model_dtype)
                h = module.register_forward_hook(self._make_intervention_hook(f_u, 1.0))
                hooks.append(h)
        
        try:
            with torch.no_grad():
                forget_loss = self.model(**forget_batch).loss.item()
                retain_loss = self.model(**retain_batch).loss.item()
        finally:
            for h in hooks:
                h.remove()
        
        result = {
            "config": config.to_dict(),
            "selected_layers": selected,
            "n_selected": len(selected),
            "mask": mask,
            "forget_loss": forget_loss,
            "retain_loss": retain_loss,
            "train_time_seconds": train_time,
            "attributions": attributions,
        }
        
        # Save
        result_path = self.output_dir / f"{config.name}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Completed {config.name}: forget={forget_loss:.4f}, retain={retain_loss:.4f}, "
                   f"selected={len(selected)}/{self.n_layers}")
        
        return result


def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", default="results/layer_selection")
    parser.add_argument("--method", default="all", choices=["all", "topk", "l1_sparse", "ste_binary", "greedy"])
    parser.add_argument("--direction_lr", type=float, default=0.01)
    parser.add_argument("--direction_steps", type=int, default=50)
    parser.add_argument("--lambda_retain", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--lambda_sparse", type=float, default=0.1)
    parser.add_argument("--compare_all", action="store_true", help="Run all methods for comparison")
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    # Load TOFU
    logger.info("Loading TOFU dataset...")
    forget_ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain90", split="train")
    
    selector = TDULayerSelector(
        model=model,
        tokenizer=tokenizer,
        forget_dataset=forget_ds,
        retain_dataset=retain_ds,
        output_dir=args.output_dir,
    )
    
    if args.compare_all:
        # Run all methods
        configs = [
            LayerSelectionConfig(method=SelectionMethod.ALL, direction_lr=args.direction_lr, 
                                direction_steps=args.direction_steps, lambda_retain=args.lambda_retain),
            LayerSelectionConfig(method=SelectionMethod.TOPK, direction_lr=args.direction_lr,
                                direction_steps=args.direction_steps, lambda_retain=args.lambda_retain,
                                top_k=3),
            LayerSelectionConfig(method=SelectionMethod.TOPK, direction_lr=args.direction_lr,
                                direction_steps=args.direction_steps, lambda_retain=args.lambda_retain,
                                top_k=5),
            LayerSelectionConfig(method=SelectionMethod.L1_SPARSE, direction_lr=args.direction_lr,
                                direction_steps=args.direction_steps, lambda_retain=args.lambda_retain,
                                lambda_sparse=0.1),
            LayerSelectionConfig(method=SelectionMethod.L1_SPARSE, direction_lr=args.direction_lr,
                                direction_steps=args.direction_steps, lambda_retain=args.lambda_retain,
                                lambda_sparse=0.5),
            LayerSelectionConfig(method=SelectionMethod.STE_BINARY, direction_lr=args.direction_lr,
                                direction_steps=args.direction_steps, lambda_retain=args.lambda_retain),
        ]
        
        results = []
        for config in configs:
            result = selector.run(config)
            results.append(result)
        
        # Summary table
        logger.info("\n" + "="*80)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*80)
        logger.info(f"{'Method':<25} {'Forget↑':<12} {'Retain↓':<12} {'Selected':<10} {'Time(s)':<10}")
        logger.info("-"*80)
        for r in results:
            name = r["config"]["method"]
            if r["config"]["method"] == "topk":
                name += f"_k{r['config']['top_k']}"
            elif r["config"]["method"] == "l1_sparse":
                name += f"_l{r['config']['lambda_sparse']}"
            logger.info(f"{name:<25} {r['forget_loss']:<12.4f} {r['retain_loss']:<12.4f} "
                       f"{r['n_selected']:<10} {r['train_time_seconds']:<10.1f}")
        
        # Save summary
        summary_path = selector.output_dir / "comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
    else:
        # Run single method
        method = SelectionMethod(args.method)
        config = LayerSelectionConfig(
            method=method,
            direction_lr=args.direction_lr,
            direction_steps=args.direction_steps,
            lambda_retain=args.lambda_retain,
            top_k=args.top_k,
            lambda_sparse=args.lambda_sparse,
        )
        selector.run(config)
    
    logger.info(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
