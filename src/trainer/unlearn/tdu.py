"""
Targeted Direction Unlearning (TDU)

Two-phase unlearning method:
1. Phase 1: Find the output direction f_u that encodes the fact to forget
2. Phase 2: Apply rank-one weight edit to erase the direction

Based on the insight that unlearning = removing specific rank-one components
from weight matrices. The edit W_new = W - f_u @ f_u.T @ W projects out the
forget direction, automatically satisfying the NSD constraint for true forgetting.

Reference: Targeted Unlearning via Concept Directions (Murphy, 2026)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm

from trainer.unlearn.grad_diff import GradDiff

import logging

logger = logging.getLogger(__name__)


class TDU(GradDiff):
    """
    Targeted Direction Unlearning.

    Finds the output direction f_u that a weight matrix writes to encode a fact,
    then surgically erases it via rank-one projection.

    Integrates with the standard training loop by overriding train() to implement
    the two-phase approach while using the repo's data infrastructure.
    """

    def __init__(
        self,
        target_modules: List[str] = None,
        direction_lr: float = 1e-2,
        direction_steps: int = 100,
        lambda_retain: float = 1.0,
        lambda_norm: float = 10.0,
        use_renormalization: bool = True,
        multi_direction: bool = False,
        max_directions: int = 3,
        direction_threshold: float = 0.1,
        num_batches: int = 10,  # Number of batches to use for direction finding
        *args,
        **kwargs,
    ):
        # Initialize without KL retain loss since we handle retain differently
        super().__init__(retain_loss_type="NLL", *args, **kwargs)

        self.target_modules = target_modules or ["mlp.down_proj", "self_attn.o_proj"]
        self.direction_lr = direction_lr
        self.direction_steps = direction_steps
        self.lambda_retain = lambda_retain
        self.lambda_norm = lambda_norm
        self.use_renormalization = use_renormalization
        self.multi_direction = multi_direction
        self.max_directions = max_directions
        self.direction_threshold = direction_threshold
        self.num_batches = num_batches

        # Ensure ref_model exists for retain KL computation
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        # Store discovered directions per layer
        self.forget_directions: Dict[str, List[torch.Tensor]] = {}

        # Learnable direction parameters (initialized per module in phase 1)
        self._direction_params: Dict[str, nn.Parameter] = {}

    def _get_target_modules(self) -> Dict[str, nn.Module]:
        """Get target weight matrices for direction finding."""
        target_dict = {}
        for name, module in self.model.named_modules():
            for target_name in self.target_modules:
                if target_name in name and isinstance(module, nn.Linear):
                    target_dict[name] = module
        return target_dict

    def _init_direction_params(self, target_modules: Dict[str, nn.Module]):
        """Initialize learnable direction parameters for each target module."""
        self._direction_params = {}
        for name, module in target_modules.items():
            hidden_size = module.out_features
            f_u = torch.randn(hidden_size, device=self.accelerator.device)
            f_u = F.normalize(f_u, dim=0)
            self._direction_params[name] = nn.Parameter(f_u)

    def _forward_with_intervention(
        self,
        inputs: Dict[str, torch.Tensor],
        target_modules: Dict[str, nn.Module],
    ) -> torch.Tensor:
        """
        Forward pass with intervention: project out f_u from each target module's output.

        Intervention: output = W @ x - f_u @ (f_u.T @ W @ x)
                            = (I - f_u @ f_u.T) @ W @ x
        """
        hooks = []

        for name, module in target_modules.items():
            f_u = self._direction_params[name]

            # Normalize (handles both renormalization and loss-based approaches)
            if self.use_renormalization:
                f_u_normalized = F.normalize(f_u, dim=0)
            else:
                f_u_normalized = f_u

            def make_hook(direction):
                def hook(mod, input, output):
                    # output shape: (batch, seq, hidden)
                    # direction shape: (hidden,)
                    d = direction.to(output.dtype)
                    projection = torch.einsum("bsh,h->bs", output, d)
                    return output - torch.einsum("bs,h->bsh", projection, d)

                return hook

            h = module.register_forward_hook(make_hook(f_u_normalized))
            hooks.append(h)

        try:
            outputs = self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        return outputs

    def _compute_direction_loss(
        self,
        forget_inputs: Dict[str, torch.Tensor],
        retain_inputs: Dict[str, torch.Tensor],
        target_modules: Dict[str, nn.Module],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the loss for direction finding.

        L = L_steer + lambda_retain * L_retain + lambda_norm * L_norm

        L_steer: We want intervention to INCREASE forget loss (break recall)
        L_retain: We want intervention to NOT change retain predictions (KL div)
        L_norm: Unit norm constraint (if not using renormalization)
        """
        # Original forget loss (without intervention)
        with torch.no_grad():
            orig_forget_loss = self.model(**forget_inputs).loss.detach()

        # Intervened forget loss
        intervened_forget_loss = self._forward_with_intervention(
            forget_inputs, target_modules
        ).loss

        # Steer loss: we want intervention to INCREASE loss (break recall)
        # So we minimize negative of (intervened - original)
        steer_loss = -(intervened_forget_loss - orig_forget_loss)

        # Retain loss: KL divergence between original and intervened
        # Only compute on the last token logits to save memory
        with torch.no_grad():
            orig_retain_logits = self.model(**retain_inputs).logits.detach()

        intervened_retain_logits = self._forward_with_intervention(
            retain_inputs, target_modules
        ).logits

        # KL divergence (intervened || original) — compute in float32 for stability
        orig_probs = F.softmax(orig_retain_logits.float(), dim=-1)
        intervened_log_probs = F.log_softmax(intervened_retain_logits.float(), dim=-1)
        retain_loss = F.kl_div(intervened_log_probs, orig_probs, reduction="batchmean")

        # Free logits early
        del orig_retain_logits, intervened_retain_logits, orig_probs, intervened_log_probs

        # Norm loss (if not using renormalization)
        norm_loss = torch.tensor(0.0, device=self.accelerator.device)
        if not self.use_renormalization:
            for name, f_u in self._direction_params.items():
                norm_loss = norm_loss + (torch.norm(f_u) - 1.0) ** 2

        total_loss = (
            steer_loss + self.lambda_retain * retain_loss + self.lambda_norm * norm_loss
        )

        metrics = {
            "steer_loss": steer_loss.item(),
            "retain_loss": retain_loss.item(),
            "norm_loss": norm_loss.item(),
            "total_loss": total_loss.item(),
            "orig_forget_loss": orig_forget_loss.item(),
            "intervened_forget_loss": intervened_forget_loss.item(),
        }

        return total_loss, metrics

    def _collect_batches(
        self, dataloader, num_batches: int
    ) -> List[Dict[str, torch.Tensor]]:
        """Collect a fixed number of batches from the dataloader."""
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batches.append(batch)
        return batches

    def _prepare_inputs_from_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Extract and prepare forget/retain inputs from a batch."""
        forget_data = batch["forget"]
        retain_data = batch["retain"]

        forget_inputs = {
            "input_ids": forget_data["input_ids"].to(self.accelerator.device),
            "attention_mask": forget_data["attention_mask"].to(self.accelerator.device),
            "labels": forget_data["labels"].to(self.accelerator.device),
        }

        retain_inputs = {
            "input_ids": retain_data["input_ids"].to(self.accelerator.device),
            "attention_mask": retain_data["attention_mask"].to(self.accelerator.device),
            "labels": retain_data["labels"].to(self.accelerator.device),
        }

        return forget_inputs, retain_inputs

    def phase1_find_directions(
        self,
        batches: List[Dict[str, Any]],
        target_modules: Dict[str, nn.Module],
    ):
        """
        Phase 1: Find forget directions for all target modules.

        Optimizes direction parameters to maximize the effect of intervention
        on forget set while minimizing effect on retain set.
        """
        logger.info("=== Phase 1: Finding forget directions ===")
        logger.info(f"Target modules: {list(target_modules.keys())}")
        logger.info(f"Using {len(batches)} batches, {self.direction_steps} steps")

        # Initialize direction parameters
        self._init_direction_params(target_modules)

        # Create optimizer for direction parameters
        optimizer = torch.optim.Adam(
            list(self._direction_params.values()), lr=self.direction_lr
        )

        # Check if wandb is available for logging
        _wandb = None
        if self.args.report_to and "wandb" in self.args.report_to:
            try:
                import wandb
                if wandb.run is not None:
                    _wandb = wandb
                else:
                    _wandb = wandb
                    _wandb.init(
                        project=os.environ.get("WANDB_PROJECT", "tdu"),
                        name=getattr(self.args, "run_name", "tdu"),
                        config={
                            "direction_lr": self.direction_lr,
                            "direction_steps": self.direction_steps,
                            "lambda_retain": self.lambda_retain,
                            "lambda_norm": self.lambda_norm,
                            "use_renormalization": self.use_renormalization,
                            "num_batches": self.num_batches,
                            "target_modules": self.target_modules,
                            "num_target_layers": len(target_modules),
                        },
                    )
            except ImportError:
                pass

        # Training loop
        self.model.eval()  # Keep model frozen during direction finding

        for step in tqdm(range(self.direction_steps), desc="Finding directions"):
            total_loss = torch.tensor(0.0, device=self.accelerator.device)
            total_metrics = {}

            optimizer.zero_grad()

            # Accumulate gradients over batches (backprop per batch to save memory)
            for batch in batches:
                forget_inputs, retain_inputs = self._prepare_inputs_from_batch(batch)

                loss, metrics = self._compute_direction_loss(
                    forget_inputs, retain_inputs, target_modules
                )
                # Scale loss and backprop immediately to free the graph
                scaled_loss = loss / len(batches)
                scaled_loss.backward()

                total_loss = total_loss + scaled_loss.detach()

                # Accumulate metrics
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v / len(batches)

            optimizer.step()
            torch.cuda.empty_cache()

            # Renormalize after step
            if self.use_renormalization:
                with torch.no_grad():
                    for name, f_u in self._direction_params.items():
                        f_u.data = F.normalize(f_u.data, dim=0)

            # Log metrics
            delta = total_metrics['intervened_forget_loss'] - total_metrics['orig_forget_loss']
            if _wandb is not None:
                _wandb.log({
                    "phase1/steer_loss": total_metrics["steer_loss"],
                    "phase1/retain_loss": total_metrics["retain_loss"],
                    "phase1/norm_loss": total_metrics["norm_loss"],
                    "phase1/total_loss": total_metrics["total_loss"],
                    "phase1/orig_forget_loss": total_metrics["orig_forget_loss"],
                    "phase1/intervened_forget_loss": total_metrics["intervened_forget_loss"],
                    "phase1/delta": delta,
                    "phase1/step": step,
                })

            if step % 20 == 0:
                logger.info(
                    f"Step {step}: "
                    f"steer={total_metrics['steer_loss']:.4f}, "
                    f"retain={total_metrics['retain_loss']:.4f}, "
                    f"delta={delta:.4f}"
                )

        # Store final directions
        for name, f_u in self._direction_params.items():
            direction = F.normalize(f_u.data.clone(), dim=0)
            self.forget_directions[name] = [direction]
            logger.info(f"Found direction for {name}")

    def phase2_apply_edits(self, target_modules: Dict[str, nn.Module]):
        """
        Phase 2: Apply weight edits to erase found directions.

        W_new = W - sum_i(f_u_i @ f_u_i.T @ W)
              = (I - sum_i(f_u_i @ f_u_i.T)) @ W
        """
        logger.info("=== Phase 2: Applying weight edits ===")

        for name, module in target_modules.items():
            if name not in self.forget_directions:
                continue

            directions = self.forget_directions[name]
            logger.info(f"Editing {name} with {len(directions)} direction(s)")

            with torch.no_grad():
                W = module.weight.data  # (out_features, in_features)

                for f_u in directions:
                    # Ensure f_u is on the same device and dtype as W
                    f_u = f_u.to(device=W.device, dtype=W.dtype)

                    # W_delta = f_u @ f_u.T @ W
                    # projection = outer(f_u, f_u) @ W
                    projection = torch.outer(f_u, f_u) @ W  # (out, in)
                    W = W - projection

                module.weight.data = W

        logger.info("Weight edits applied")

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Override train() to implement the two-phase TDU approach.

        1. Collect batches from the dataloader
        2. Phase 1: Find forget directions
        3. Phase 2: Apply weight edits
        4. Save the model
        """
        logger.info("=== Starting TDU Unlearning ===")

        # Get target modules
        target_modules = self._get_target_modules()
        if not target_modules:
            raise ValueError(
                f"No target modules found matching patterns: {self.target_modules}"
            )
        logger.info(f"Found {len(target_modules)} target modules")

        # Get dataloader
        train_dataloader = self.get_train_dataloader()

        # Collect batches for direction finding
        logger.info(f"Collecting {self.num_batches} batches for direction finding...")
        batches = self._collect_batches(train_dataloader, self.num_batches)
        if len(batches) < self.num_batches:
            logger.warning(
                f"Only collected {len(batches)} batches "
                f"(requested {self.num_batches})"
            )

        # Phase 1: Find directions
        self.phase1_find_directions(batches, target_modules)

        # Phase 2: Apply edits
        self.phase2_apply_edits(target_modules)

        # Save model
        output_dir = self.args.output_dir
        logger.info(f"Saving model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        self.save_model(output_dir)

        # Run evaluation if configured
        if self.args.do_eval and self.evaluators:
            logger.info("Running evaluation...")
            self.evaluate()

        logger.info("=== TDU Unlearning Complete ===")

        return None  # No TrainOutput since we don't use standard training loop

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Standard compute_loss for evaluation compatibility.

        TDU doesn't use this during training - it overrides train() instead.
        This method is here for evaluation with the HF Trainer.
        """
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        outputs = model(**forget_inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
