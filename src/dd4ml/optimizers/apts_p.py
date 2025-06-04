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
        self.loc_model = mark_trainable(clone_model(model))

        # Instantiate local optimizer (trust‐region or LSSR1_TR)
        self.loc_optim = local_opt(
            self.loc_model.parameters(), **local_opt_params
        )
        
        self.loc_closure = self.non_foc_loc_closure


    @torch.no_grad()
    def sync_loc_to_glob(self) -> None:
        """
        Make the global model identical on every rank by merging the
        sub-domain (trainable) updates held in `self.loc_model`.

        Assumption: `mark_trainable` assigned disjoint index sets, i.e.
        each parameter tensor is owned by exactly one rank.
        """
        # -------- single-process fallback ------------------------------------
        if (
            not (dist.is_available() and dist.is_initialized())
            or dist.get_world_size() == 1
        ):
            with torch.no_grad():
                for pg, pl in zip(
                    self.model.parameters(), self.loc_model.parameters()
                ):
                    if pl.requires_grad:  # this rank owns it
                        pg.copy_(pl.data)
            self.loc_model.load_state_dict(self.model.state_dict())
            return

        # -------- multi-process merge ----------------------------------------
        with torch.no_grad():
            # 1. Copy owned slices; zero the others
            for pg, pl in zip(self.model.parameters(), self.loc_model.parameters()):
                if pl.requires_grad:
                    pg.copy_(pl.data)  # owner rank writes its update
                else:
                    pg.zero_()  # non-owners write zeros

            # 2. Sum across all ranks – only the owner contributes a non-zero value
            for pg in self.model.parameters():
                dist.all_reduce(pg.data, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def loc_grad_to_vector(self):
        return trainable_params_to_vector(self.loc_model)

    def step(self, inputs, labels):
        # Store inputs and labels for closures
        self.inputs, self.labels = inputs, labels
        
        # Reset gradient evaluation counters (as Python floats)
        # Note: closures will increment these
        self.grad_evals = 0.0
        self.loc_grad_evals = 0.0  # track local grad evals as Python int

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()
    
        # Compute initial global/local loss and gradient
        self.init_glob_loss, self.init_loc_loss = self.glob_closure(compute_grad=True), = self.loc_closure(compute_grad=True)

        # Store initial global/local gradients (flattened)
        self.init_glob_grad, self.init_loc_grad = self.glob_grad_to_vector(), self.loc_grad_to_vector()
        
        # Perform local optimization steps
        loc_loss, _ = self.loc_steps(self.init_loc_loss, self.init_loc_grad)
        
        # Account for local gradient evaluations across all models
        self.grad_evals += self.loc_grad_evals * self.nr_models
    
        # Synchronize parameters from local models to global model
        self.sync_loc_to_glob()

        # Compute trial step and ensure it is within trust region
        step = self.glob_params_to_vector() - self.init_glob_flat
        step = self.ensure_step_within_tr(step)
        
        pred = None
        if not self.dogleg: 
            # Aggregate local losses
            pred = loc_loss - self.init_loc_loss
            if self.nr_models > 1:
                dist.all_reduce(pred, op=dist.ReduceOp.SUM)
                
        # Else, pred will be computed as second-order approximation 
        loss, grad, self.glob_optim.delta  = self.control_step(step, pred)
            
        # Optional global pass
        if self.global_pass:
            loss, grad = self.glob_steps(loss, grad)
        
        # Synchronize global and local models and set delta accordingly
        self.sync_glob_to_loc()
        
        return loss
