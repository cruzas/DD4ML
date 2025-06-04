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

        # Clone model for local updates; avoids overwriting global params
        self.loc_model = clone_model(model)

        # Instantiate local optimiser (trust-region or LSSR1_TR)
        self.loc_opt = loc_opt(
            self.loc_model.parameters(), **loc_opt_hparams
        )
        
        # Choose local closure based on first-order correction flag
        self.loc_closure = (
            self.foc_loc_closure if self.foc else self.non_foc_loc_closure
        )
        
        # Print name of glob_opt and loc_opt
        dprint(f"APTS_P global optimizer: {self.glob_opt.__name__}; local optimizer: {self.loc_opt.__name__}")

    @torch.no_grad()
    def aggregate_loc_steps_and_losses(self, step, loc_red):
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
        return step, loc_red
        
    def step(self, inputs, labels):
        """
        Performs one APTS_D step: evaluate initial losses/gradients,
        run local iterations, propose a step, test acceptance, and possibly
        run additional global iterations.
        """
        # Store inputs and labels for closures
        self.inputs, self.labels = inputs, labels
        
        # Reset gradient evaluation counters (as Python floats)
        # Note: closures will increment these
        self.grad_evals, self.loc_grad_evals = 0.0, 0.0 

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.glob_params_to_vector()
        
        # Compute initial global/local loss and gradient
        self.init_glob_loss, self.init_loc_loss = self.glob_closure(compute_grad=True), = self.loc_closure(compute_grad=True)

        # Store initial global/local gradients (flattened)
        self.init_glob_grad, self.init_loc_grad = self.glob_grad_to_vector(), self.loc_grad_to_vector()
            
        # Calculate residual between global and local gradients
        self.resid = self.init_glob_grad - self.init_loc_grad
        
        # Perform local optimization steps
        loc_loss, _ = self.loc_steps(self.init_loc_loss, self.init_loc_grad)
        
        # Account for local gradient evaluations across all models
        self.grad_evals += self.loc_grad_evals * self.nr_models

        # Compute local step and reduction, then aggregate across all models:
        # step becomes global trial step
        with torch.no_grad():
            step = self.loc_params_to_vector() - self.init_glob_flat
            loc_red = self.init_loc_loss - loc_loss
        step, pred = self.aggregate_loc_steps_and_losses(step, loc_red)
        
        # Ensure step is within trust region
        step = self.ensure_step_within_tr(step)

        # APTS trust-region control: possibly modifies self.delta and global model parameters
        loss, grad, self.glob_opt.delta = self.control_step(step, pred)        

        # Optional global pass
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad)

        # Synchronize global and local models and set delta accordingly
        self.sync_glob_to_loc()
           
        return loss
