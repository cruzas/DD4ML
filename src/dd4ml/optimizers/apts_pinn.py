import torch
import torch.distributed as dist

from .apts_d import APTS_D


class APTS_PINN(APTS_D):
    __name__ = "APTS_PINN"

    def __init__(
        self,
        *args,
        num_subdomains: int = 1,
        overlap: float = 0.0,
        criterion=None,
        **kwargs,
    ):
        super().__init__(*args, nr_models=num_subdomains, criterion=criterion, **kwargs)
        self.num_subdomains = max(1, int(num_subdomains))
        low = getattr(self.criterion, "low", 0.0)
        high = getattr(self.criterion, "high", 1.0)
        self.domain_low, self.domain_high = float(low), float(high)
        self.subdomain_bounds = torch.linspace(low, high, self.num_subdomains + 1)
        self.overlap = max(float(overlap), 0.0)

        # Cache for efficiency
        self._bounds_cache = {}  # Cache computed subdomain bounds
        self._half_overlap = self.overlap / 2.0  # Pre-compute half overlap
        self._criterion_attrs = ["current_xt", "current_x"]  # Attributes to check/set

    def get_subdomain_bounds(self, idx: int):
        """Return (low, high) bounds for a subdomain with optional overlap."""
        # Cache bounds to avoid repeated tensor item() calls
        if idx not in self._bounds_cache:
            base_low = self.subdomain_bounds[idx].item()
            base_high = self.subdomain_bounds[idx + 1].item()
            if self.overlap > 0.0:
                low = max(base_low - self._half_overlap, self.domain_low)
                high = min(base_high + self._half_overlap, self.domain_high)
                self._bounds_cache[idx] = (low, high)
            else:
                self._bounds_cache[idx] = (base_low, base_high)
        return self._bounds_cache[idx]

    def get_subdomain_mask(self, x: torch.Tensor, idx: int):
        """Build a boolean mask selecting points for subdomain ``idx``."""
        low, high = self.get_subdomain_bounds(idx)
        return (x >= low) & (x <= high)

    def _set_criterion_input(self, input_tensor):
        """Helper to set criterion attributes if they exist."""
        for attr in self._criterion_attrs:
            if hasattr(self.criterion, attr):
                setattr(self.criterion, attr, input_tensor)

    def _prepare_tensor(self, tensor):
        """Helper to prepare tensor with gradient tracking."""
        return tensor.detach().requires_grad_(True) if tensor is not None else None

    def step(self, inputs, labels, inputs_d=None, labels_d=None, hNk=None):
        # Initialize the global and local gradient evaluations counters
        self.grad_evals = self.loc_grad_evals = 0.0

        # Build mask for the subdomain
        idx = dist.get_rank() if dist.is_initialized() else 0
        # ``inputs`` may have more than one feature (e.g., space and time).
        # The partitioning into subdomains is done only along the spatial
        # dimension, which is assumed to be stored in the first column.
        # Using ``squeeze()`` on the full tensor previously produced a mask with
        # shape ``[N, F]`` (where ``F`` is the number of features). When this
        # mask was applied to ``labels`` with shape ``[N, 1]`` it triggered an
        # ``IndexError`` because the mask's second dimension did not match.
        # Extract the first feature so that the mask is one-dimensional and can
        # be safely used to index all tensors along the batch dimension.
        x = inputs[..., 0].squeeze()
        mask = self.get_subdomain_mask(x, idx)

        # Create tensors for full domain (for global pass) and subdomain (for local pass)
        full_in = self._prepare_tensor(inputs)
        full_lab = labels
        full_in_d = self._prepare_tensor(inputs_d)
        full_lab_d = labels_d

        sub_in = full_in[mask]
        sub_lab = full_lab[mask] if full_lab is not None else None
        sub_in_d = full_in_d[mask] if full_in_d is not None else None
        sub_lab_d = full_lab_d[mask] if full_lab_d is not None else None

        n_local = int(sub_in.shape[0]) if sub_in is not None else 0
        N_total = int(full_in.shape[0])
        weight = n_local / max(N_total, 1)

        self.inputs, self.labels = full_in, full_lab
        self.inputs_d, self.labels_d = full_in_d, full_lab_d
        self.hNk = hNk

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()

        # Compute the initial global loss and gradient
        self._set_criterion_input(full_in)
        self.init_glob_loss = self.glob_closure_main(compute_grad=True)
        self.init_glob_grad = self.glob_grad_to_vector()

        # Compute the initial local loss and gradient
        self.inputs, self.labels = sub_in, sub_lab
        self.inputs_d, self.labels_d = sub_in_d, sub_lab_d
        self._set_criterion_input(sub_in)

        self.init_loc_loss = self.loc_closure(compute_grad=True)
        self.init_loc_grad = self.loc_grad_to_vector()

        # Compute the residual if foc is enabled
        if self.foc:
            self.resid = self.init_glob_grad - self.init_loc_grad

        # Perform local optimization steps
        loc_loss, _ = self.loc_steps(self.init_loc_loss, self.init_loc_grad)

        # Compute the local step and reduction
        with torch.no_grad():
            step = self.loc_params_to_vector() - self.init_glob_flat
            loc_red = self.init_loc_loss - loc_loss

        step, pred = self.aggregate_loc_steps_and_losses(step, loc_red, weight=weight)

        # APTS trust-region control: possibly modifies self.delta and global model parameters
        self.inputs, self.labels = full_in, full_lab
        self.inputs_d, self.labels_d = full_in_d, full_lab_d
        self._set_criterion_input(full_in)
        loss, grad, self.glob_opt.delta = self.control_step(step, pred)
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad)

        # ── Sync for next iteration ──
        self.sync_glob_to_loc()
        return loss
