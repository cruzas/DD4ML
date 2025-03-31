import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .utils import get_trust_region_params


class TrustRegionEMA(torch.optim.Optimizer):
    @staticmethod
    def setup_TR_args(config):
        params = get_trust_region_params(config)
        config.max_iter = params["max_iter"]
        config.lr = params["lr"]
        config.max_lr = params["max_lr"]
        config.min_lr = params["min_lr"]
        config.nu = params["nu"]
        config.inc_factor = params["inc_factor"]
        config.dec_factor = params["dec_factor"]
        config.nu_1 = params["nu_1"]
        config.nu_2 = params["nu_2"]
        config.norm_type = params["norm_type"]
        return config

    def __init__(
        self,
        model,
        lr=0.01,
        max_lr=1.0,
        min_lr=0.0001,
        nu=0.5,
        inc_factor=2.0,
        dec_factor=0.5,
        nu_1=0.25,
        nu_2=0.75,
        max_iter=5,
        norm_type=2,
        trust_ratio_beta=0.9,  # Smoothing factor for EMA.
    ):
        super().__init__(
            model.parameters(),
            {"lr": lr, "max_lr": max_lr, "min_lr": min_lr, "max_iter": max_iter},
        )
        self.model = model
        self.param_list = list(model.parameters())
        self.lr = lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        self.nu = min(nu, nu_1)
        self.max_iter = max_iter
        self.norm_type = norm_type
        # Initialize the EMA of the reduction ratio.
        self.trust_ratio_ema = None
        self.trust_ratio_beta = trust_ratio_beta

    def _apply_update_vectorized(self, grad, scale):
        with torch.no_grad():
            params_vector = parameters_to_vector(self.param_list)
            updated_vector = params_vector - scale * grad
            vector_to_parameters(updated_vector, self.param_list)

    def step(self, closure, old_loss=None, grad=None):
        # Compute initial loss if not provided.
        if old_loss is None:
            old_loss = closure(compute_grad=True)

        # Cache current parameters.
        old_params = parameters_to_vector(self.param_list).clone()

        # Process gradients.
        if grad is None:
            grad = parameters_to_vector(
                [p.grad.detach() for p in self.param_list if p.grad is not None]
            )
        elif hasattr(grad, "tensor"):
            grad = parameters_to_vector(grad.tensor)

        grad_norm = grad.norm(p=self.norm_type)
        if grad_norm <= torch.finfo(torch.float32).eps:
            print(f"Stopping TrustRegion due to ||g|| = {grad_norm}.")
            return old_loss

        # Compute the update scale with current learning rate.
        scale = self.lr / grad_norm
        self._apply_update_vectorized(grad, scale)
        candidate_loss = closure(compute_grad=False)

        # Compute the predicted reduction.
        pred_red = self.lr * grad_norm
        act_red = old_loss - candidate_loss
        red_ratio = act_red / pred_red if pred_red != 0 else 0.0

        # Update the EMA of the reduction ratio.
        if self.trust_ratio_ema is None:
            self.trust_ratio_ema = red_ratio
        else:
            self.trust_ratio_ema = (
                self.trust_ratio_beta * self.trust_ratio_ema
                + (1 - self.trust_ratio_beta) * red_ratio
            )

        # Adjust self.lr based on the EMA.
        if self.trust_ratio_ema < self.nu_1:
            self.lr = max(self.min_lr, self.dec_factor * self.lr)
        elif self.trust_ratio_ema > self.nu_2:
            self.lr = min(self.max_lr, self.inc_factor * self.lr)

        # Accept the candidate update if it decreases the loss.
        if candidate_loss <= old_loss:
            return candidate_loss
        else:
            # Revert parameters if loss did not decrease.
            vector_to_parameters(old_params, self.param_list)
            return old_loss
