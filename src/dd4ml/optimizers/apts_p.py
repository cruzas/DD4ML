from .apts_base import *


class APTS_P(APTS_Base):
    __name__ = "APTS_P"

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
        norm_type=math.inf,
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
            norm_type=math.inf,
            max_loc_iters=max_loc_iters,
            max_glob_iters=max_glob_iters,
            tol=tol,
        )

        # Clone model for local updates; avoids overwriting global params
        self.loc_model = mark_trainable(clone_model(model))

        # Trainable parameters
        filtered = []
        for g in self.loc_model.parameters():
            if isinstance(g, dict):
                grp = dict(g)
                grp["params"] = [p for p in g["params"] if p.requires_grad]
                if grp["params"]:
                    filtered.append(grp)
            else:
                if g.requires_grad:
                    filtered.append(g)

        # Instantiate local optimizer (trust‐region or LSSR1_TR)
        self.loc_opt = loc_opt(
            params=filtered,
            flat_grads_fn=self.loc_grad_to_vector,
            flat_params_fn=self.loc_params_to_vector,
            **loc_opt_hparams,
        )

        self.loc_closure = self.non_foc_loc_closure
        if isinstance(self.loc_opt, ASNTR):
            self.loc_closure_d = self.non_foc_loc_closure_d
        
        # Compute number of parameters in local and global models
        self.n_global = self.glob_params_to_vector().numel()
        self.n_local = self.loc_params_to_vector().numel()
        self.sqrt_n_global = math.sqrt(self.n_global)
        self.sqrt_n_local = math.sqrt(self.n_local)
        
        # Modify delta for global and local optimizers
        self.glob_opt.delta = min(self.glob_opt.max_delta, self.delta * self.sqrt_n_global)
        self.loc_opt.delta = min(self.loc_opt.max_delta, self.delta * self.sqrt_n_local)

        # Print name of glob_opt and loc_opt
        dprint(
            f"{self.__name__} global optimizer: {type(self.glob_opt).__name__}; local optimizer: {type(self.loc_opt).__name__}"
        )

    @torch.no_grad()
    def sync_loc_to_glob(self) -> None:
        """
        Make the global model identical on every rank by merging the
        sub-domain (trainable) updates held in ``self.loc_model".

        Assumption: ``mark_trainable" assigned disjoint index sets, i.e.
        each parameter tensor is owned by exactly one rank.
        """
        # Single-process fallback
        if (
            not (dist.is_available() and dist.is_initialized())
            or dist.get_world_size() == 1
        ):
            with torch.no_grad():
                for pg, pl in zip(self.model.parameters(), self.loc_model.parameters()):
                    if pl.requires_grad:  # this rank owns it
                        pg.copy_(pl.data)
            self.loc_model.load_state_dict(self.model.state_dict())
            return

        # Multi-process merge
        with torch.no_grad():
            # Copy owned slices; zero the others
            for pg, pl in zip(self.model.parameters(), self.loc_model.parameters()):
                if pl.requires_grad:
                    pg.copy_(pl.data)  # owner rank writes its update
                else:
                    pg.zero_()  # non-owners write zeros

            # Sum across all ranks - only the owner contributes a non-zero value
            for pg in self.model.parameters():
                dist.all_reduce(pg.data, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def loc_params_to_vector(self):
        return trainable_params_to_vector(self.loc_model)

    @torch.no_grad()
    def loc_grad_to_vector(self):
        """
        Returns the local model's gradients as a single flat vector.
        This is used by the local optimizer to compute the step.
        """
        return trainable_grads_to_vector(self.loc_model)

    @torch.no_grad()
    def sync_glob_to_loc(self):
        self.delta = max(self.glob_opt.min_delta, self.glob_opt.delta / self.sqrt_n_global)
        self.update_pytorch_lr()

        self.loc_opt.delta = min(self.loc_opt.max_delta, self.delta * self.sqrt_n_local)
        if hasattr(self.loc_opt, "update_pytorch_lr"):
            self.loc_opt.update_pytorch_lr()

        # Ensure local model matches global model for the next iteration
        self.loc_model.load_state_dict(get_state_dict(self.model))

    def step(self, inputs, labels, inputs_d=None, labels_d=None, hNk=None):
        """
        Performs one APTS_P step: evaluate initial losses/gradients,
        run local iterations, propose a step, test acceptance, and possibly
        run additional global iterations.

        inputs_d, labels_d are only used in case ASNTR is the global or local optimizer.
        """

        # Store inputs and labels for closures
        self.inputs, self.labels = inputs, labels
        self.inputs_d, self.labels_d = inputs_d, labels_d
        self.hNk = hNk

        # Reset gradient evaluation counters (as Python floats).
        # Note: closures will increment these.
        self.grad_evals, self.loc_grad_evals = 0.0, 0.0

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()

        # Compute initial global and local losses and gradients
        self.init_glob_loss, self.init_loc_loss = self.glob_closure_main(
            compute_grad=True
        ), self.loc_closure(compute_grad=True)

        # Store initial global/local gradients (flattened)
        self.init_glob_grad, self.init_loc_grad = (
            self.glob_grad_to_vector(),
            self.loc_grad_to_vector(),
        )

        # Perform local optimization steps
        loc_loss, _ = self.loc_steps(self.init_loc_loss, self.init_loc_grad)

        # Synchronize parameters from local models to global model
        self.sync_loc_to_glob()

        # Compute trial step
        step = self.glob_params_to_vector() - self.init_glob_flat

        # Aggregate local losses
        pred = self.init_loc_loss - loc_loss
        if self.nr_models > 1:
            dist.all_reduce(pred, op=dist.ReduceOp.SUM)

        # Perform global control on step and update delta
        loss, grad, new_base_delta = self.control_step(step, pred)
        self.delta = new_base_delta

        # Update global optimizer's delta based on the new base delta    
        self.glob_opt.delta = min(self.max_delta, new_base_delta * self.sqrt_n_global)

        # Optional global pass
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad)
        
        # Synchronize global and local models and set delta accordingly
        self.sync_glob_to_loc()

        return loss
