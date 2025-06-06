from .apts_base import *


class APTS_IP(APTS_Base):
    __name__ = "APTS_IP"

    @staticmethod
    def setup_APTS_hparams(config):
        loc_opts = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        try:
            config.loc_opt = loc_opts[config.loc_opt.lower()]
        except KeyError:
            raise ValueError(f"Unknown subdomain optimizer: {config.loc_opt}")

        config.loc_opt_hparams = {"lr": config.learning_rate}
        if config.loc_opt in {torch.optim.Adam, torch.optim.AdamW}:
            config.loc_opt_hparams["betas"] = config.betas
        elif config.loc_opt == torch.optim.SGD:
            config.loc_opt_hparams["momentum"] = 0.9

        config.glob_opt, config.glob_opt_hparams = (
            (LSSR1_TR, get_lssr1_tr_hparams(config))
            if config.glob_second_order
            else (TR, get_tr_hparams(config))
        )
        # Disable gradient broadcast in the global optimizer as each rank
        # holds only a shard of the model when running APTS_IP.
        config.glob_opt_hparams["sync"] = False
        config.apts_params = get_apts_params(config)
        return config

    def __init__(
        self,
        params,
        model=None,
        delta=0.01,
        min_delta=None,
        max_delta=None,
        nu_dec=None,
        nu_inc=None,
        inc_factor=None,
        dec_factor=None,
        glob_opt=None,
        glob_opt_hparams=None,
        loc_opt=None,
        loc_opt_hparams=None,
        *,
        glob_pass=True,
        norm_type=2,
        max_loc_iters=0,
        max_glob_iters=3,
        dogleg=False,
        tol=1e-6,
        APTS_in_data_sync_strategy="average",
        step_strategy="mean",
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

        # Synchronize non-parameter attributes from the first param group.
        self._sync_attributes_from_param_group()
        self.APTS_in_data_sync_strategy = APTS_in_data_sync_strategy.lower()
        self.step_strategy = step_strategy

        if self.step_strategy not in ["weighted_mean", "mean"]:
            raise ValueError(
                'The step strategy must be either "weighted_mean" or "mean".'
            )
        if self.APTS_in_data_sync_strategy not in ["average", "sum"]:
            raise ValueError(
                'The APTS in data synchronization strategy must be either "average" or "sum".'
            )
        if self.APTS_in_data_sync_strategy == "sum" and dist.get_rank() == 0:
            print(
                '(WARNING) APTS in data "sum" synchronization strategy still has to be tested/verified.'
            )

        self.loc_opt = loc_opt(params=model.subdomain_params(), **loc_opt_hparams)
        
        glob_opt_hparams["flat_params"] = self.model.parameters()
        self.glob_opt = glob_opt(
            params=list(model.parameters()), **glob_opt_hparams
        )
        self.glob_opt._flat_grads_fn = self.model.grad
        self.glob_opt._flat_params_fn = self.model.parameters

        # Print name of glob_opt and loc_opt
        dprint(
            f"{self.__name__} global optimizer: {type(self.glob_opt).__name__}; local optimizer: {type(self.loc_opt).__name__}"
        )

    def loc_steps(self, final_subdomain_closure=None):
        for i in range(self.max_loc_iters):
            self.loc_opt.zero_grad()
            self.loc_opt.step()
            if i != self.max_loc_iters - 1:
                outputs = self.model.subdomain_forward()
                losses = (
                    final_subdomain_closure(outputs)
                    if self.model.model_handler.is_last_stage()
                    else []
                )
                self.model.subdomain_backward(losses)
        self.model.sync_params(method="average")
        self._update_param_group()

    def step(self, closure, final_subdomain_closure):
        # Reset gradient evaluation counters (as Python floats)
        self.grad_evals, self.loc_grad_evals = 0.0, 0.0

        # Save initial global parameters (flattened, cloned to avoid in-place)
        self.init_glob_flat = self.model.parameters(clone=True)

        # Compute initial global/local loss and gradient
        self.init_glob_loss = closure(compute_grad=True, zero_grad=True)
        self.init_glob_grad = self.model.grad(clone=True)
        
        print(f"Rank {dist.get_rank()}. Initial global rank dimension: {self.init_glob_grad.dim()}")
        print(f"Rank {dist.get_rank()}. Detached initial global grad shape: {self.init_glob_grad.detach().shape}")
        
        self.grad_evals += 1

        # Perform local steps
        self.loc_steps(final_subdomain_closure)

        # Compute global trial step
        step = self.model.parameters(clone=False) - self.init_glob_flat

        # APTS trust-region control: possibly modifies self.delta and global model parameters
        flat_grads_fn = self.model.grad
        loss, grad, self.glob_opt.delta = self.control_step(step, closure=closure)

        # Optional global pass
        if self.glob_pass:
            loss, grad = self.glob_steps(loss, grad, closure=closure)

        self.delta = self.glob_opt.delta
        self._update_param_group()

        return loss
