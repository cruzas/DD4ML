from .apts_base import *

class APTS_P(APTS_Base):
    __name__ = "APTS_P"

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
        dogleg=False,
        norm_type=2,
        max_local_iters=3,
        max_global_iters=3,
        tol=1e-6,
    ):
        # Call base for shared defaults, buffer, device, global optimizer
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

        # Subclass‐specific state
        self.dogleg = bool(dogleg)

        # Clone model for local updates; avoids overwriting global params
        self.local_model = mark_trainable(clone_model(model))

        # Instantiate local optimiser (trust‐region or LSSR1_TR)
        if "LSSR1_TR" in str(local_opt):
            self.local_optimizer = local_opt(
                self.local_model.parameters(), **local_opt_params
            )
        else:
            self.local_optimizer = local_opt(
                self.local_model, **local_opt_params
            )

    @torch.no_grad()
    def sync_loc_to_glob(self) -> None:
        """
        Make the global model identical on every rank by merging the
        sub-domain (trainable) updates held in `self.local_model`.

        Assumption: `mark_trainable` assigned *disjoint* index sets, i.e.
        each parameter tensor is owned by exactly one rank.
        """
        # -------- single-process fallback ------------------------------------
        if (
            not (dist.is_available() and dist.is_initialized())
            or dist.get_world_size() == 1
        ):
            with torch.no_grad():
                for pg, pl in zip(
                    self.model.parameters(), self.local_model.parameters()
                ):
                    if pl.requires_grad:  # this rank owns it
                        pg.copy_(pl.data)
            self.local_model.load_state_dict(self.model.state_dict())
            return

        # -------- multi-process merge ----------------------------------------
        with torch.no_grad():
            # 1. Copy owned slices; zero the others
            for pg, pl in zip(self.model.parameters(), self.local_model.parameters()):
                if pl.requires_grad:
                    pg.copy_(pl.data)  # owner rank writes its update
                else:
                    pg.zero_()  # non-owners write zeros

            # 2. Sum across all ranks – only the owner contributes a non-zero value
            for pg in self.model.parameters():
                dist.all_reduce(pg.data, op=dist.ReduceOp.SUM)

    def step(self, inputs, labels):
        """
        Performs one APTS_P step: evaluate initial losses/gradients,
        run local iterations, propose a step, test acceptance, and possibly
        run additional global iterations.
        """
        hp = self.defaults # hyper-parameters
        
        # ------------------------------------------------------------------ #
        # Book-keeping and initial evaluations
        # ------------------------------------------------------------------ #
        # Reset gradient evaluation counter (as Python float)
        self.grad_evals = 0.0
        self.loc_grad_evals = 0.0  # track local grad evals as Python int
        
        # Store inputs and labels for closures
        self.inputs, self.labels = inputs, labels
        # Choose local closure (only one option for APTS_P)
        loc_closure = self.non_foc_loc_closure

        # Save initial global parameters (flattened, cloned to avoid in-place)
        with torch.no_grad():
            init_glob_flat = flatten_params(self.model, self._flat_params_buffer).clone()

        # Compute initial global loss and gradient
        init_glob_loss = self.global_closure(compute_grad=True)
        # Count one global gradient eval
        self.grad_evals += 1.0
        
        # Compute initial local loss and gradient
        init_local_loss = self.local_closure(compute_grad=True)
        self.loc_grad_evals += 1

        # Store initial gradients (flattened)
        with torch.no_grad():
            init_glob_grad = parameters_to_vector(
                [p.grad for p in self.model.parameters()]
            ).detach()
            init_loc_grad = trainable_parameters_to_vector(self.local_model)
        
        # Perform local optimisation steps
        loc_loss, loc_grad = self.local_steps(init_local_loss, init_loc_grad)
        # Account for local gradient evaluations across all models
        self.grad_evals += self.loc_grad_evals * self.nr_models
    
        # Synchronize parameters from local models to global model
        self.sync_loc_to_glob()

        # Compute step: current global params minus initial global params
        curr_flat = flatten_params(self.model, self._local_flat_buffer)
        step = curr_flat - init_glob_flat

        # ------------------------------------------------------------------ #
        # Step correction / aggregation
        # ------------------------------------------------------------------ #
        if not self.dogleg:  # TR control
            with torch.no_grad():
                # Ensure loc_loss is a tensor to allow arithmetic
                loc_loss = ensure_tensor(loc_loss, device=self.device)
                # Reduction in local loss
                loc_red = init_loc_loss - loc_loss.detach()
                
                # If more than one model, loc_red is the sum of all local reductions
                if self.nr_models > 1:
                    dist.all_reduce(loc_red, op=dist.ReduceOp.SUM)
                
                # Ensure step norm is within trust region
                step_norm = step.norm(p=hp["norm_type"])
                if step_norm > hp["delta"]:
                    print("Warning: step norm exceeds delta, scaling down.")
                    step = (hp["delta"] / step_norm) * step

                # Apply the candidate step
                restore_params(self.model, step)
                
            trial_loss = self.global_closure()

            actual_reduction = init_glob_loss - trial_loss
            predicted_reduction = torch.dot(init_glob_grad, step) - (0.5 * step_norm**2)
            rho = actual_reduction / predicted_reduction

            if rho < 0.25:
                # Too small step, reduce the step size
                self.delta = max(
                    self.delta * self.glob_optim.dec_factor,
                    self.glob_optim.min_delta,
                )
                restore_params(self.model, -step)
                trial_loss = init_glob_loss
            elif rho > 0.75:
                # Good step, increase the step size
                self.delta = min(
                    self.delta * self.glob_optim.inc_factor,
                    self.glob_optim.max_delta,
                )
                restore_params(self.model, step)

                
        else:
            delta = self.defaults["delta"]
            w = 0
            restore_params(self.model, init_glob_flat + step)
            trial_loss = self.global_closure()
            while trial_loss > init_glob_loss and w <= 1:
                with torch.no_grad():
                    delta *= self.glob_optim.dec_factor
                    w += 0.2
                    step_update = ((1 - w) * step) - (w * init_glob_grad)
                    step_update = (delta / step_update.norm()) * step_update
                    restore_params(self.model, step_update)
                trial_loss = self.global_closure()
                torch.cuda.empty_cache()
        
        loss = self.glob_optim.step(
            closure=self.global_closure, old_loss=trial_loss
        )
        # Local models get global model parameters, as does the local optimizer
        with torch.no_grad():
            self.delta = self.glob_optim.delta
            self.local_optimizer.delta = self.delta
            self.local_model.load_state_dict(self.model.state_dict())
        return loss.item()
