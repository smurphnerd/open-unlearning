"""
TDU Hyperparameter Sweep with Post-hoc Top-k Layer Selection

Sweeps over:
- direction_lr: learning rate for direction optimization
- direction_steps: number of optimization steps
- lambda_retain: weight on retain loss
- k: number of top layers to keep (post-hoc selection)

Evaluates on TOFU forget10 split with Llama-3.2-1B-Instruct.
"""

import os
import json
import torch
import itertools
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""
    direction_lr: float
    direction_steps: int
    lambda_retain: float
    top_k: int  # -1 means all layers
    
    # Fixed params
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    target_modules: str = "down_proj"  # or "dense_4h_to_h" for Pythia
    use_renormalization: bool = True
    lambda_norm: float = 10.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def name(self) -> str:
        k_str = "all" if self.top_k == -1 else f"top{self.top_k}"
        return f"lr{self.direction_lr}_steps{self.direction_steps}_lret{self.lambda_retain}_{k_str}"


@dataclass
class SweepResult:
    """Results from a single sweep run."""
    config: SweepConfig
    
    # Per-layer attribution scores (for analysis)
    layer_attributions: Dict[str, float]
    selected_layers: List[str]
    
    # Metrics before layer selection
    forget_loss_all: float
    retain_loss_all: float
    
    # Metrics after top-k selection
    forget_loss_topk: float
    retain_loss_topk: float
    
    # Timing
    train_time_seconds: float
    
    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "layer_attributions": self.layer_attributions,
            "selected_layers": self.selected_layers,
            "forget_loss_all": self.forget_loss_all,
            "retain_loss_all": self.retain_loss_all,
            "forget_loss_topk": self.forget_loss_topk,
            "retain_loss_topk": self.retain_loss_topk,
            "train_time_seconds": self.train_time_seconds,
        }


class TDUSweeper:
    """
    Run TDU hyperparameter sweep with post-hoc layer selection.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        forget_dataset,
        retain_dataset,
        device: str = "cuda",
        output_dir: str = "results/sweep",
        target_pattern: str = None,  # Auto-detect if None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect target pattern based on model architecture
        if target_pattern is None:
            target_pattern = self._detect_target_pattern()
        
        # Find target modules
        self.target_modules = self._find_target_modules(target_pattern)
        self.n_layers = len(self.target_modules)
        logger.info(f"Found {self.n_layers} target modules matching '{target_pattern}'")
    
    def _detect_target_pattern(self) -> str:
        """Auto-detect the MLP down projection pattern based on model architecture."""
        # Check for common patterns
        all_names = [name for name, _ in self.model.named_modules()]
        all_names_str = " ".join(all_names)
        
        if "down_proj" in all_names_str:
            return "down_proj"  # Llama, Mistral, etc.
        elif "dense_4h_to_h" in all_names_str:
            return "dense_4h_to_h"  # Pythia, GPT-NeoX
        elif "c_proj" in all_names_str:
            return "c_proj"  # GPT-2
        elif "fc2" in all_names_str:
            return "fc2"  # Some other architectures
        else:
            logger.warning("Could not auto-detect target pattern, defaulting to 'down_proj'")
            return "down_proj"
    
    def _find_target_modules(self, pattern: str) -> Dict[str, torch.nn.Linear]:
        modules = {}
        for name, module in self.model.named_modules():
            if pattern in name and isinstance(module, torch.nn.Linear):
                modules[name] = module
        return modules
    
    def _prepare_batch(self, dataset, max_samples: int = 16) -> Dict[str, torch.Tensor]:
        """Prepare a batch from dataset."""
        samples = dataset.select(range(min(len(dataset), max_samples)))
        
        # Tokenize
        texts = [s["text"] if "text" in s else s["question"] + " " + s["answer"] 
                 for s in samples]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        encoded["labels"] = encoded["input_ids"].clone()
        
        return encoded
    
    def _train_directions(
        self,
        config: SweepConfig,
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train f_u directions for all layers.
        Returns (directions_dict, attribution_scores).
        """
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Initialize directions in fp32 for stability (even if model is fp16)
        directions = {}
        for name, module in self.target_modules.items():
            hidden_size = module.out_features
            f_u = torch.randn(hidden_size, device=self.device, dtype=torch.float32)
            f_u = F.normalize(f_u, dim=0)
            directions[name] = nn.Parameter(f_u)
        
        optimizer = torch.optim.Adam(directions.values(), lr=config.direction_lr)
        
        # Track gradients for attribution
        grad_norms = {name: [] for name in directions}
        
        for step in range(config.direction_steps):
            optimizer.zero_grad()
            
            # Normalize directions and get model dtype for hooks
            model_dtype = next(self.model.parameters()).dtype
            normalized = {name: F.normalize(d, dim=0) for name, d in directions.items()}
            
            # Forward with intervention on all layers
            hooks = []
            for name, module in self.target_modules.items():
                f_u = normalized[name].to(model_dtype)  # Cast to model dtype for hook
                
                def make_hook(direction):
                    def hook(mod, inp, out):
                        proj = torch.einsum('bsh,h->bs', out, direction)
                        return out - torch.einsum('bs,h->bsh', proj, direction)
                    return hook
                
                h = module.register_forward_hook(make_hook(f_u))
                hooks.append(h)
            
            # Compute forget loss (we want intervention to INCREASE this)
            try:
                outputs_intervened = self.model(**forget_batch)
                forget_loss_intervened = outputs_intervened.loss
            finally:
                for h in hooks:
                    h.remove()
            
            # Original forget loss (no intervention)
            with torch.no_grad():
                outputs_orig = self.model(**forget_batch)
                forget_loss_orig = outputs_orig.loss
            
            # Steer loss: want intervention to break recall
            steer_loss = -(forget_loss_intervened - forget_loss_orig)
            
            # DEBUG: Check for NaN sources
            if step > 0 and step % 10 == 0:
                logger.info(f"  DEBUG step {step}: forget_int={forget_loss_intervened.item():.4f}, "
                           f"forget_orig={forget_loss_orig.item():.4f}, steer={steer_loss.item():.4f}")
            
            # Retain loss: KL divergence on retain set
            hooks = []
            for name, module in self.target_modules.items():
                f_u = normalized[name].to(model_dtype)  # Cast to model dtype for hook
                
                def make_hook(direction):
                    def hook(mod, inp, out):
                        proj = torch.einsum('bsh,h->bs', out, direction)
                        return out - torch.einsum('bs,h->bsh', proj, direction)
                    return hook
                
                h = module.register_forward_hook(make_hook(f_u))
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
            
            # DEBUG: Check for NaN in retain_kl and directions
            if step > 0 and step % 10 == 0:
                max_logit_int = outputs_retain_int.logits.abs().max().item()
                max_logit_orig = outputs_retain_orig.logits.abs().max().item()
                dir_norms = [d.norm().item() for d in directions.values()]
                logger.info(f"  DEBUG step {step}: retain_kl={retain_kl.item():.4f}, "
                           f"max_logit_int={max_logit_int:.1f}, max_logit_orig={max_logit_orig:.1f}, "
                           f"dir_norm_range=[{min(dir_norms):.3f}, {max(dir_norms):.3f}]")
            
            # Total loss
            total_loss = steer_loss + config.lambda_retain * retain_kl
            
            # Early NaN detection
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"  NaN/Inf detected at step {step}!")
                logger.warning(f"    steer_loss={steer_loss.item()}, retain_kl={retain_kl.item()}")
                logger.warning(f"    forget_int={forget_loss_intervened.item()}, forget_orig={forget_loss_orig.item()}")
                max_logit = outputs_retain_int.logits.abs().max().item()
                logger.warning(f"    max_logit_intervened={max_logit}")
                break  # Stop this config early
            
            total_loss.backward()
            
            # Record gradient norms for attribution
            max_grad = 0
            for name, param in directions.items():
                if param.grad is not None:
                    gn = param.grad.norm().item()
                    grad_norms[name].append(gn)
                    max_grad = max(max_grad, gn)
            
            # Debug: log gradient magnitude
            if step == 0:
                logger.info(f"  Step 0 max_grad_norm: {max_grad:.2f}")
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(directions.values(), max_norm=1.0)
            
            optimizer.step()
            
            # Renormalize
            if config.use_renormalization:
                with torch.no_grad():
                    for name in directions:
                        directions[name].data = F.normalize(directions[name].data, dim=0)
            
            if step % 20 == 0:
                logger.info(f"Step {step}: steer={steer_loss.item():.4f}, retain={retain_kl.item():.4f}")
        
        # Compute attribution scores (mean gradient norm)
        attributions = {name: np.mean(norms) for name, norms in grad_norms.items()}
        
        # Detach directions
        final_directions = {name: d.detach().clone() for name, d in directions.items()}
        
        return final_directions, attributions
    
    def _evaluate_with_selection(
        self,
        directions: Dict[str, torch.Tensor],
        selected_layers: List[str],
        forget_batch: Dict[str, torch.Tensor],
        retain_batch: Dict[str, torch.Tensor],
    ) -> Tuple[float, float]:
        """
        Evaluate with only selected layers' interventions active.
        Returns (forget_loss, retain_loss).
        """
        import torch.nn.functional as F
        
        # Register hooks only for selected layers
        model_dtype = next(self.model.parameters()).dtype
        hooks = []
        for name in selected_layers:
            module = self.target_modules[name]
            f_u = directions[name].to(model_dtype)  # Cast to model dtype
            
            def make_hook(direction):
                def hook(mod, inp, out):
                    proj = torch.einsum('bsh,h->bs', out, direction)
                    return out - torch.einsum('bs,h->bsh', proj, direction)
                return hook
            
            h = module.register_forward_hook(make_hook(f_u))
            hooks.append(h)
        
        try:
            with torch.no_grad():
                forget_out = self.model(**forget_batch)
                retain_out = self.model(**retain_batch)
        finally:
            for h in hooks:
                h.remove()
        
        return forget_out.loss.item(), retain_out.loss.item()
    
    def _select_top_k_layers(
        self,
        attributions: Dict[str, float],
        k: int,
    ) -> List[str]:
        """Select top-k layers by attribution score."""
        if k == -1 or k >= len(attributions):
            return list(attributions.keys())
        
        sorted_layers = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_layers[:k]]
    
    def run_single(self, config: SweepConfig) -> SweepResult:
        """Run a single sweep configuration."""
        import time
        
        logger.info(f"Running config: {config.name}")
        
        # Prepare batches
        forget_batch = self._prepare_batch(self.forget_dataset)
        retain_batch = self._prepare_batch(self.retain_dataset)
        
        # Train directions
        start_time = time.time()
        directions, attributions = self._train_directions(config, forget_batch, retain_batch)
        train_time = time.time() - start_time
        
        # Evaluate with all layers
        all_layers = list(directions.keys())
        forget_loss_all, retain_loss_all = self._evaluate_with_selection(
            directions, all_layers, forget_batch, retain_batch
        )
        
        # Select top-k and evaluate
        selected_layers = self._select_top_k_layers(attributions, config.top_k)
        forget_loss_topk, retain_loss_topk = self._evaluate_with_selection(
            directions, selected_layers, forget_batch, retain_batch
        )
        
        result = SweepResult(
            config=config,
            layer_attributions=attributions,
            selected_layers=selected_layers,
            forget_loss_all=forget_loss_all,
            retain_loss_all=retain_loss_all,
            forget_loss_topk=forget_loss_topk,
            retain_loss_topk=retain_loss_topk,
            train_time_seconds=train_time,
        )
        
        # Save individual result
        result_path = self.output_dir / f"{config.name}.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Completed {config.name}: forget_all={forget_loss_all:.4f}, "
                   f"forget_topk={forget_loss_topk:.4f}, selected={len(selected_layers)} layers")
        
        return result
    
    def run_sweep(
        self,
        direction_lrs: List[float] = [0.001, 0.01, 0.1],
        direction_steps: List[int] = [50, 100, 200],
        lambda_retains: List[float] = [0.1, 1.0, 10.0],
        top_ks: List[int] = [1, 2, 3, 5, -1],  # -1 means all
    ) -> List[SweepResult]:
        """Run full hyperparameter sweep."""
        
        configs = []
        for lr, steps, lret, k in itertools.product(
            direction_lrs, direction_steps, lambda_retains, top_ks
        ):
            configs.append(SweepConfig(
                direction_lr=lr,
                direction_steps=steps,
                lambda_retain=lret,
                top_k=k,
            ))
        
        logger.info(f"Running sweep with {len(configs)} configurations")
        
        results = []
        for config in tqdm(configs, desc="Sweep"):
            try:
                result = self.run_single(config)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on {config.name}: {e}")
                continue
        
        # Save summary
        summary_path = self.output_dir / "sweep_summary.json"
        with open(summary_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Find best config
        best_by_forget = min(results, key=lambda r: -r.forget_loss_topk)  # Higher is better for forget
        best_balanced = min(results, key=lambda r: -r.forget_loss_topk + r.retain_loss_topk)
        
        logger.info(f"\nBest by forget loss: {best_by_forget.config.name}")
        logger.info(f"  forget={best_by_forget.forget_loss_topk:.4f}, retain={best_by_forget.retain_loss_topk:.4f}")
        logger.info(f"\nBest balanced: {best_balanced.config.name}")
        logger.info(f"  forget={best_balanced.forget_loss_topk:.4f}, retain={best_balanced.retain_loss_topk:.4f}")
        
        return results


def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", default="results/tdu_sweep")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced grid")
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
    
    # Load TOFU dataset
    logger.info("Loading TOFU dataset...")
    forget_ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain90", split="train")
    
    # Create sweeper
    sweeper = TDUSweeper(
        model=model,
        tokenizer=tokenizer,
        forget_dataset=forget_ds,
        retain_dataset=retain_ds,
        output_dir=args.output_dir,
    )
    
    # Run sweep
    if args.quick:
        # Quick test with small grid
        results = sweeper.run_sweep(
            direction_lrs=[0.01],
            direction_steps=[50],
            lambda_retains=[1.0],
            top_ks=[1, 3, -1],
        )
    else:
        # Full sweep
        results = sweeper.run_sweep()
    
    logger.info(f"\nSweep complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
