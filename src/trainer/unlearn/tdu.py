import torch
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_

from trainer.unlearn.grad_diff import GradDiff


class TDU(GradDiff):
    """
    Targeted Direction Unlearning (TDU)

    Learns output directions f_u that encode facts to forget, then projects them out
    via rank-one edits: W_θ x = Wx - f_u(f_u^T Wx) = (I - f_u f_u^T) Wx

    This satisfies the Negative Semi-Definite (NSD) constraint automatically when ||f_u|| = 1,
    guaranteeing true forgetting.
    """

    def __init__(self, hook_points=["o_proj", "down_proj"], *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        # Freeze all model parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.hooks = []

        # Attach target directions and hooks directly to Linear modules
        for name, module in self.model.named_modules():
            for hook_point in hook_points:
                if name.endswith(hook_point) and hasattr(module, "weight"):
                    # f_u: learnable direction [1, out_features]
                    hidden_size = module.weight.shape[0]  # out_features
                    td = Parameter(
                        xavier_uniform_(
                            torch.empty(1, hidden_size, device=module.weight.device)
                        )
                    )
                    setattr(module, "target_direction", td)

                    # Hook implements: output = Wx - f_u(f_u^T Wx)
                    hook = module.register_forward_hook(self._create_hook())
                    self.hooks.append(hook)

    def _create_hook(self):
        """
        Intervention: W_θ x = Wx - f_u(f_u^T Wx) = (I - f_u f_u^T) Wx
        Projects out the target direction from the output.
        """

        def hook(module, input, output):
            f_u = module.target_direction  # [1, hidden_size]

            # Normalize to unit vector (required for NSD guarantee)
            f_u_norm = f_u / f_u.norm(dim=-1, keepdim=True)

            # output: [batch, seq_len, hidden_size]
            # Projection: f_u(f_u^T @ output) = (output @ f_u.T) @ f_u
            projection = (output @ f_u_norm.T) @ f_u_norm

            return output - projection

        return hook

    def compute_loss(self, model, inputs, return_outputs=False):
        # L_steer: maximize forget loss (model can't recall)
        # L_retain: minimize KL divergence on retain set
        # Inherited from GradDiff: gamma * (-forget_loss) + alpha * retain_loss
        return super().compute_loss(model, inputs, return_outputs)

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
