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

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from trainer.unlearn.base import UnlearnTrainer

import logging

logger = logging.getLogger(__name__)


class TDU(UnlearnTrainer):
    """
    Targeted Direction Unlearning.
    
    Finds the output direction f_u that a weight matrix writes to encode a fact,
    then surgically erases it via rank-one projection.
    """
    
    def __init__(
        self,
        target_modules: List[str] = None,  # e.g., ["mlp.up_proj", "mlp.down_proj"]
        direction_lr: float = 1e-2,
        direction_steps: int = 100,
        lambda_retain: float = 1.0,
        lambda_norm: float = 10.0,
        use_renormalization: bool = True,  # vs loss-based norm constraint
        multi_direction: bool = False,  # find multiple orthogonal directions
        max_directions: int = 3,
        direction_threshold: float = 0.1,  # min loss improvement for additional direction
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.target_modules = target_modules or ["mlp.down_proj"]
        self.direction_lr = direction_lr
        self.direction_steps = direction_steps
        self.lambda_retain = lambda_retain
        self.lambda_norm = lambda_norm
        self.use_renormalization = use_renormalization
        self.multi_direction = multi_direction
        self.max_directions = max_directions
        self.direction_threshold = direction_threshold
        
        # Store reference model for comparison
        self.ref_model = self._prepare_ref_model(self.model)
        
        # Store discovered directions per layer
        self.forget_directions: Dict[str, List[torch.Tensor]] = {}
        
    def _prepare_ref_model(self, model):
        """Create frozen copy of model for reference."""
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = ref_model.to(self.accelerator.device)
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model
    
    def _get_target_modules(self, model) -> Dict[str, nn.Module]:
        """Get target weight matrices for direction finding."""
        target_dict = {}
        for name, module in model.named_modules():
            for target_name in self.target_modules:
                if target_name in name and isinstance(module, nn.Linear):
                    target_dict[name] = module
        return target_dict
    
    def _forward_with_intervention(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        module_name: str,
        module: nn.Linear,
        f_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with intervention: project out f_u from module's output.
        
        Intervention: output = W @ x - f_u @ (f_u.T @ W @ x)
                            = (I - f_u @ f_u.T) @ W @ x
        """
        hooks = []
        
        def intervention_hook(mod, input, output):
            # output shape: (batch, seq, hidden)
            # f_u shape: (hidden,)
            # Project out f_u component
            projection = torch.einsum('bsh,h->bs', output, f_u)
            output_intervened = output - torch.einsum('bs,h->bsh', projection, f_u)
            return output_intervened
        
        hook = module.register_forward_hook(intervention_hook)
        hooks.append(hook)
        
        try:
            outputs = model(**inputs)
        finally:
            for h in hooks:
                h.remove()
                
        return outputs
    
    def _compute_steer_loss(
        self,
        model,
        forget_inputs: Dict[str, torch.Tensor],
        module_name: str,
        module: nn.Linear,
        f_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute steering loss: how much does projecting out f_u break fact recall?
        
        L_steer = -log p(y_forget | x) + log p(y_forget | x, intervention)
        
        We want this to be HIGH (intervention breaks recall).
        """
        # Original loss (without intervention)
        with torch.no_grad():
            orig_outputs = model(**forget_inputs)
            orig_loss = orig_outputs.loss
        
        # Intervened loss
        intervened_outputs = self._forward_with_intervention(
            model, forget_inputs, module_name, module, f_u
        )
        intervened_loss = intervened_outputs.loss
        
        # Steer loss: we want intervention to INCREASE loss (break recall)
        # So we minimize negative of (intervened - original)
        steer_loss = -(intervened_loss - orig_loss)
        
        return steer_loss
    
    def _compute_retain_loss(
        self,
        model,
        retain_inputs: Dict[str, torch.Tensor],
        module_name: str,
        module: nn.Linear,
        f_u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute retain loss: KL divergence between original and intervened on retain set.
        
        We want this to be LOW (intervention doesn't affect retain set).
        """
        with torch.no_grad():
            orig_outputs = model(**retain_inputs)
            orig_logits = orig_outputs.logits
        
        intervened_outputs = self._forward_with_intervention(
            model, retain_inputs, module_name, module, f_u
        )
        intervened_logits = intervened_outputs.logits
        
        # KL divergence
        orig_probs = F.softmax(orig_logits, dim=-1)
        intervened_log_probs = F.log_softmax(intervened_logits, dim=-1)
        
        kl_div = F.kl_div(intervened_log_probs, orig_probs, reduction='batchmean')
        
        return kl_div
    
    def _find_direction(
        self,
        model,
        forget_inputs: Dict[str, torch.Tensor],
        retain_inputs: Dict[str, torch.Tensor],
        module_name: str,
        module: nn.Linear,
        existing_directions: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Find the output direction f_u that encodes the fact to forget.
        
        Optimizes: L = L_steer + lambda_retain * L_retain + lambda_norm * L_norm
        
        Returns: (direction, final_loss)
        """
        existing_directions = existing_directions or []
        
        # Initialize direction randomly
        hidden_size = module.out_features
        f_u = torch.randn(hidden_size, device=self.accelerator.device)
        f_u = F.normalize(f_u, dim=0)
        f_u = nn.Parameter(f_u)
        
        optimizer = torch.optim.Adam([f_u], lr=self.direction_lr)
        
        best_loss = float('inf')
        best_direction = f_u.data.clone()
        
        for step in range(self.direction_steps):
            optimizer.zero_grad()
            
            # Normalize if using renormalization
            if self.use_renormalization:
                f_u_normalized = F.normalize(f_u, dim=0)
            else:
                f_u_normalized = f_u
            
            # Gram-Schmidt orthogonalization against existing directions
            for existing in existing_directions:
                proj = torch.dot(f_u_normalized, existing) * existing
                f_u_normalized = f_u_normalized - proj
                f_u_normalized = F.normalize(f_u_normalized, dim=0)
            
            # Compute losses
            steer_loss = self._compute_steer_loss(
                model, forget_inputs, module_name, module, f_u_normalized
            )
            
            retain_loss = self._compute_retain_loss(
                model, retain_inputs, module_name, module, f_u_normalized
            )
            
            # Norm loss (if not using renormalization)
            if not self.use_renormalization:
                norm_loss = (torch.norm(f_u) - 1.0) ** 2
            else:
                norm_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            total_loss = (
                steer_loss 
                + self.lambda_retain * retain_loss 
                + self.lambda_norm * norm_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
            # Renormalize after step
            if self.use_renormalization:
                with torch.no_grad():
                    f_u.data = F.normalize(f_u.data, dim=0)
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_direction = f_u.data.clone()
                
            if step % 20 == 0:
                logger.info(
                    f"[{module_name}] Step {step}: "
                    f"steer={steer_loss.item():.4f}, "
                    f"retain={retain_loss.item():.4f}, "
                    f"total={total_loss.item():.4f}"
                )
        
        # Final normalization
        best_direction = F.normalize(best_direction, dim=0)
        
        return best_direction, best_loss
    
    def _find_all_directions(
        self,
        model,
        forget_inputs: Dict[str, torch.Tensor],
        retain_inputs: Dict[str, torch.Tensor],
        module_name: str,
        module: nn.Linear,
    ) -> List[torch.Tensor]:
        """
        Find all directions needed to erase the fact (handles multi-dimensional concepts).
        """
        directions = []
        prev_loss = float('inf')
        
        for i in range(self.max_directions if self.multi_direction else 1):
            direction, loss = self._find_direction(
                model, forget_inputs, retain_inputs,
                module_name, module, directions
            )
            
            # Check if this direction provides sufficient improvement
            if i > 0 and (prev_loss - loss) < self.direction_threshold:
                logger.info(
                    f"[{module_name}] Stopping at {i} directions "
                    f"(improvement {prev_loss - loss:.4f} < threshold)"
                )
                break
            
            directions.append(direction)
            prev_loss = loss
            logger.info(f"[{module_name}] Found direction {i+1}, loss={loss:.4f}")
        
        return directions
    
    def _apply_weight_edit(
        self,
        module: nn.Linear,
        directions: List[torch.Tensor],
    ):
        """
        Apply rank-one weight edit to erase directions.
        
        W_new = W - sum_i(f_u_i @ f_u_i.T @ W)
              = (I - sum_i(f_u_i @ f_u_i.T)) @ W
        """
        with torch.no_grad():
            W = module.weight.data  # (out_features, in_features)
            
            for f_u in directions:
                # W_delta = f_u @ (f_u.T @ W) = outer(f_u, W.T @ f_u)
                # But W is (out, in), so f_u.T @ W doesn't work directly
                # We need: (f_u @ f_u.T) @ W where f_u is (out,)
                # This is: outer(f_u, f_u) @ W = f_u[:, None] @ f_u[None, :] @ W
                projection = torch.outer(f_u, f_u) @ W  # (out, in)
                W = W - projection
            
            module.weight.data = W
    
    def phase1_find_directions(
        self,
        forget_inputs: Dict[str, torch.Tensor],
        retain_inputs: Dict[str, torch.Tensor],
    ):
        """
        Phase 1: Find forget directions for all target modules.
        """
        logger.info("=== Phase 1: Finding forget directions ===")
        
        target_modules = self._get_target_modules(self.model)
        
        for module_name, module in target_modules.items():
            logger.info(f"Processing {module_name}...")
            
            directions = self._find_all_directions(
                self.model, forget_inputs, retain_inputs,
                module_name, module
            )
            
            self.forget_directions[module_name] = directions
            logger.info(f"Found {len(directions)} direction(s) for {module_name}")
    
    def phase2_apply_edits(self):
        """
        Phase 2: Apply weight edits to erase found directions.
        """
        logger.info("=== Phase 2: Applying weight edits ===")
        
        target_modules = self._get_target_modules(self.model)
        
        for module_name, module in target_modules.items():
            if module_name in self.forget_directions:
                directions = self.forget_directions[module_name]
                logger.info(f"Editing {module_name} with {len(directions)} direction(s)...")
                self._apply_weight_edit(module, directions)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Standard compute_loss for compatibility with trainer loop.
        
        TDU doesn't use the standard training loop - instead call:
        1. phase1_find_directions()
        2. phase2_apply_edits()
        
        This method is here for evaluation compatibility.
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
    
    def unlearn(
        self,
        forget_inputs: Dict[str, torch.Tensor],
        retain_inputs: Dict[str, torch.Tensor],
    ):
        """
        Main entry point: run both phases to unlearn.
        """
        self.phase1_find_directions(forget_inputs, retain_inputs)
        self.phase2_apply_edits()
        logger.info("=== Unlearning complete ===")
