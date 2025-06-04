from .apts_base import *

class APTS_D(APTS_Base):
    __name__ = "APTS_D"
    
    @staticmethod
    def setup_APTS_args(config):
        return APTS_Base.setup_APTS_args(config)

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
        global_opt=None,
        global_opt_params=None,
        local_opt=None,
        local_opt_params=None,
        *,
        global_pass=True,
        foc=True,
        norm_type=2,
        max_local_iters=3,
        max_global_iters=3,
        tol=1e-6,
    ):
        # Call base for shared defaults (only 'lr'), buffer, device, global optimiser
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
            global_opt=global_opt,
            global_opt_params=global_opt_params,
            local_opt=local_opt,
            local_opt_params=local_opt_params,
            global_pass=global_pass,
            norm_type=norm_type,
            max_local_iters=max_local_iters,
            max_global_iters=max_global_iters,
            tol=tol,
        )

        # Subclass-specific state
        self.foc = bool(foc)

        # Clone model for local updates; avoids overwriting global params
        self.loc_model = clone_model(model)

        # Instantiate local optimiser (trust-region or LSSR1_TR)
        self.loc_optim = local_opt(
            self.loc_model.parameters(), **local_opt_params
        )
        
        # Print name of glob_optim and loc_optim
        dprint(f"APTS_P global optimizer: {self.glob_optim.__name__}; local optimizer: {self.loc_optim.__name__}")

        
    def step(self, inputs, labels):
        """
        Performs one APTS_D step: evaluate initial losses/gradients,
        run local iterations, propose a step, test acceptance, and possibly
        run additional global iterations.
        """
        # Reset gradient evaluation counter (as Python float)
        self.grad_evals = 0.0
        self.loc_grad_evals = 0  # track local grad evals as Python int

        # Store inputs and labels for closures
        self.inputs, self.labels = inputs, labels
        # Choose local closure based on first-order correction flag
        loc_closure = (
            self.foc_loc_closure if self.foc else self.non_foc_loc_closure
        )

        # Save initial global parameters (flattened, cloned to avoid in-place)
        with torch.no_grad():
            self.init_glob_flat = flatten_params(self.model, self._flat_params_buffer).clone()

        # Compute initial global loss and gradient
        self.init_glob_loss = self.glob_closure(compute_grad=True)
        # Count one global gradient eval
        self.grad_evals += 1.0

        # Compute initial local loss and gradient
        init_loc_loss = loc_closure(compute_grad=True)
        self.loc_grad_evals += 1

        # Store initial gradients (flattened)
        self.init_glob_grad = self.glob_grad_to_vector()
        init_loc_grad = self.loc_grad_to_vector()
            
        # Calculate residual between global and local gradients
        self.resid = self.init_glob_grad - init_loc_grad
        
        # Perform local optimisation steps
        loc_loss, loc_grad = self.loc_steps(init_loc_loss, init_loc_grad, loc_closure)
        # Account for local gradient evaluations across all models
        self.grad_evals += self.loc_grad_evals * self.nr_models

        # Compute step: current local params minus initial global params
        curr_flat = flatten_params(self.loc_model, self._loc_flat_buffer)
        step = curr_flat - self.init_glob_flat

        # Step correction / aggregation
        with torch.no_grad():
            # Ensure loc_loss is a tensor to allow arithmetic
            loc_loss = ensure_tensor(loc_loss, device=self.device)
            # Reduction in local loss
            loc_red = init_loc_loss - loc_loss.detach()

            # If more than one model, global step is sum of local steps 
            # and loc_red is the sum of all local reductions
            if self.nr_models > 1:
                # Coalesce step and loc_red into one tensor
                coalesced = torch.cat([step.view(-1), loc_red.view(1)])
                # Single all_reduce on that combined tensor
                dist.all_reduce(coalesced, op=dist.ReduceOp.SUM)
                # Split back into step and loc_red
                numel = step.numel()
                step.copy_(coalesced[:numel].view_as(step))
                loc_red = coalesced[numel].unsqueeze(0)
                if self.norm_type == math.inf:
                    step /= self.nr_models
                loc_red /= self.nr_models

            # Ensure step norm is within trust region
            # step_norm = step.norm(p=self.norm_type)
            # if step_norm > self.delta:
            #     print(f"Warning: step norm {step_norm:.6e} exceeds delta {self.delta:.4f}. Difference is {step_norm - self.delta:.6e}. Scaling down step.")
            #     step = (self.delta / step_norm) * step

        # TR control
        loss, grad = self.apts_tr_control(step, loc_red)        

        # Keep global optimiser's delta in sync
        self.glob_optim.delta = self.delta

        # Optional global pass
        if self.global_pass:
            loss, grad = self.glob_steps(loss, grad)

        # Global to local synchronisation
        with torch.no_grad():
            # Sync final delta and update local optimiser accordingly
            self.delta = self.glob_optim.delta
            self.update_pytorch_lr()

            self.loc_optim.delta = self.glob_optim.delta
            if self.norm_type != math.inf and self.nr_models > 1:
                self.loc_optim.delta /= self.nr_models

            # Ensure local model matches global model for next iteration
            self.loc_model.load_state_dict(get_state_dict(self.model))

        return loss
