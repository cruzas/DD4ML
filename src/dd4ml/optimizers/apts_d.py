from .apts_base import *


class APTS_D(APTS_Base):
    __name__ = "APTS_D"

    @staticmethod
    def setup_APTS_hparams(config):
        return APTS_Base.setup_APTS_hparams(config)

    def __init__(
        self,
        params,
        model=None,
        delta=None,
        min_delta=None,
        max_delta=None,
        nu_dec=None,
        nu_inc=None,
        inc_factor=None,
        dec_factor=None,
        criterion=None,
        device=None,
        nr_models=None,
        glob_opt=None,
        glob_opt_hparams=None,
        loc_opt=None,
        loc_opt_hparams=None,
        *,
        glob_pass=True,
        foc=True,
        soc=False,
        norm_type=2,
        max_loc_iters=3,
        max_glob_iters=3,
        tol=1e-6,
    ):
        super().__init__(
            params,
            model=model,
            delta=delta,
            min_delta=min_delta,
            max_delta=max_delta,
            nu_dec=nu_dec,
            nu_inc=nu_inc,
            inc_factor=inc_factor,
            dec_factor=dec_factor,
            criterion=criterion,
            device=device,
            nr_models=nr_models,
            glob_opt=glob_opt,
            glob_opt_hparams=glob_opt_hparams,
            loc_opt=loc_opt,
            loc_opt_hparams=loc_opt_hparams,
            glob_pass=glob_pass,
            norm_type=norm_type,
            max_loc_iters=max_loc_iters,
            max_glob_iters=max_glob_iters,
            tol=tol,
        )

        # Subclass-specific state
        self.foc = bool(foc)
        self.soc = bool(soc)

        # Clone model for local updates; avoids overwriting global params
        self.loc_model = clone_model(model)

        # Instantiate local optimizer (trust-region or LSSR1_TR)
        self.loc_opt = loc_opt(self.loc_model.parameters(), **loc_opt_hparams)

        # Choose local closure based on first-order correction flag
        self.loc_closure = (
            self.foc_loc_closure if self.foc else self.non_foc_loc_closure
        )
        if isinstance(self.loc_opt, ASNTR):
            self.loc_closure_d = (
                self.foc_loc_closure_d if self.foc else self.non_foc_loc_closure_d
            )

        # Cache parameter counts to avoid expensive vector computations
        temp_global = self.glob_params_to_vector()
        temp_local = self.loc_params_to_vector()
        self.n_global = temp_global.numel()
        self.n_local = temp_local.numel()
        
        # Only compute square roots if needed
        if self.norm_type == math.inf:
            self.sqrt_n_global = math.sqrt(self.n_global)
            self.sqrt_n_local = math.sqrt(self.n_local)
            # Modify delta for global and local optimizers
            self.glob_opt.min_delta = self.min_delta * self.sqrt_n_global
            self.glob_opt.max_delta = self.max_delta * self.sqrt_n_global
            self.loc_opt.min_delta = self.min_delta * self.sqrt_n_local
            self.loc_opt.max_delta = self.max_delta * self.sqrt_n_local

            self.glob_opt.delta = min(
                self.glob_opt.max_delta, self.delta * self.sqrt_n_global
            )
            self.loc_opt.delta = min(
                self.loc_opt.max_delta, self.delta * self.sqrt_n_local
            )

        # Print name of glob_opt and loc_opt
        dprint(
            f"{self.__name__} global optimizer: {type(self.glob_opt).__name__}; local optimizer: {type(self.loc_opt).__name__}"
        )

        # Pre-allocate buffers for efficiency
        self._coalesced_buffer = None  # For aggregate_loc_steps_and_losses
        self._weight_tensor_cache = {}  # Cache for weight tensors
        self._step_buffer = torch.empty_like(temp_global)  # For step computation

    @torch.no_grad()
    def aggregate_loc_steps_and_losses(self, step, loc_red, *, weight: float = 1.0):
        # Cache weight tensors to avoid repeated tensor creation
        cache_key = (step.device, step.dtype, weight)
        if cache_key not in self._weight_tensor_cache:
            self._weight_tensor_cache[cache_key] = torch.tensor(
                float(weight), device=step.device, dtype=step.dtype
            )
        w = self._weight_tensor_cache[cache_key]
        
        # If more than one model, global step is sum of local steps
        # and loc_red is the sum of all local reductions
        if self.nr_models > 1:
            # Pre-allocate coalesced buffer on first use
            coalesced_size = step.numel() + 1
            if (self._coalesced_buffer is None or 
                self._coalesced_buffer.numel() < coalesced_size or
                self._coalesced_buffer.device != step.device or
                self._coalesced_buffer.dtype != step.dtype):
                self._coalesced_buffer = torch.empty(
                    coalesced_size, device=step.device, dtype=step.dtype
                )
            
            # Efficiently populate coalesced buffer
            numel = step.numel()
            coalesced = self._coalesced_buffer[:coalesced_size]
            coalesced[:numel] = step.view(-1)
            coalesced[numel] = loc_red * w
            
            # Single all_reduce on that combined tensor
            dist.all_reduce(coalesced, op=dist.ReduceOp.SUM)
            
            # Split back into step and loc_red
            step.copy_(coalesced[:numel].view_as(step))
            loc_red = coalesced[numel].unsqueeze(0)
            if self.norm_type == math.inf:
                step /= self.nr_models
        return step, loc_red

    def step(self, inputs, labels, inputs_d=None, labels_d=None, hNk=None):
        """
        Performs one APTS_D step: evaluate initial losses/gradients,
        run local iterations, propose a step, test acceptance, and possibly
        run additional global iterations.

        inputs_d, labels_d are only used in case we are using ASNTR as the global/local optimizer.
        """
        # Store inputs and labels for closures
        self.inputs, self.labels = inputs, labels
        self.inputs_d, self.labels_d = inputs_d, labels_d
        self.hNk = hNk

        # Reset gradient evaluation counters (as Python floats)
        # Note: closures will increment these
        self.grad_evals, self.loc_grad_evals = 0.0, 0.0

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()

        # Compute initial global/local loss and gradient
        # Note: Both closures need compute_grad=True, so can't avoid dual calls
        self.init_glob_loss = self.glob_closure_main(compute_grad=True)
        self.init_loc_loss = self.loc_closure(compute_grad=True)

        # Store initial global/local gradients (flattened)
        self.init_glob_grad, self.init_loc_grad = (
            self.glob_grad_to_vector(),
            self.loc_grad_to_vector(),
        )

        # Calculate residual between global and local gradients
        if self.foc:
            self.resid = (self.init_glob_grad - self.init_loc_grad).to(dtype=self.init_glob_grad.dtype)

        # Perform local optimization steps
        loc_loss, _ = self.loc_steps(self.init_loc_loss, self.init_loc_grad)

        # Compute local step and reduction, then aggregate across all models:
        # step becomes global trial step
        with torch.no_grad():
            # Use pre-allocated buffer for step computation
            loc_params = self.loc_params_to_vector()
            torch.sub(loc_params, self.init_glob_flat, out=self._step_buffer)
            loc_red = self.init_loc_loss - loc_loss
        step, pred = self.aggregate_loc_steps_and_losses(self._step_buffer, loc_red)

        # APTS trust-region control: possibly modifies self.delta and global model parameters
        loss, grad, self.glob_opt.delta = self.control_step(step, pred)

        # Optional global pass
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad)

        # Synchronize global and local models and set delta accordingly
        self.sync_glob_to_loc()

        return loss
