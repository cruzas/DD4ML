import torch


class TrustRegion(torch.optim.Optimizer):
    @staticmethod
    def setup_TR_args(config):
        config.max_iter = 3
        config.lr = config.learning_rate
        config.max_lr = 10.0
        config.min_lr = 1e-4
        config.nu = 0.5
        config.inc_factor = 2.0
        config.dec_factor = 0.5
        config.nu_1 = 0.25
        config.nu_2 = 0.75
        config.norm_type = 2
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
    ):
        super().__init__(
            model.parameters(),
            {"lr": lr, "max_lr": max_lr, "min_lr": min_lr, "max_iter": max_iter},
        )
        self.model = model
        self.param_list = list(model.parameters())  # Cache parameters.
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

    def _apply_update(self, grad, scale):
        with torch.no_grad():
            if not hasattr(self.model, "grad"):
                # Update using slices from the concatenated gradient.
                offset = 0
                for p in self.param_list:
                    if p.grad is not None:
                        numel = p.numel()
                        update_slice = grad[offset : offset + numel].view_as(p)
                        p.data.sub_(update_slice * scale)
                        offset += numel
            else:
                # Assume model.grad() returns a structure with one gradient per parameter.
                for i, p in enumerate(self.param_list):
                    if hasattr(grad, "tensor"):
                        p.data.sub_(grad.tensor[i] * scale)
                    else:
                        p.data.sub_(grad[i] * scale)

    def step(self, closure):
        old_loss = closure(compute_grad=True)

        # Retrieve the gradient vector.
        if not hasattr(self.model, "grad"):
            grad = torch.cat(
                [
                    p.grad.detach().view(-1)
                    for p in self.param_list
                    if p.grad is not None
                ]
            )
        else:
            grad = self.model.grad()

        # Compute the global gradient norm.
        if self.norm_type == 2:
            grad_norm = grad.norm(p=2)
        else:
            grad_norm = grad.norm(p=self.norm_type)

        if grad_norm <= torch.finfo(torch.float32).eps:
            print(f"Stopping TrustRegion algorithm due to ||g|| = {grad_norm}.")
            return old_loss

        scale = self.lr / grad_norm
        self._apply_update(grad, scale)

        new_loss = closure(compute_grad=False)
        c = 0
        while old_loss - new_loss < 0 and c < self.max_iter:
            stop = abs(self.lr - self.min_lr) / self.min_lr < 1e-6
            old_lr = self.lr

            act_red = old_loss - new_loss
            pred_red = self.lr * (
                grad.norm(p=2) if self.norm_type == 2 else grad.norm(p=self.norm_type)
            )
            red_ratio = act_red / pred_red

            if red_ratio < self.nu_1:
                self.lr = max(self.min_lr, self.dec_factor * self.lr)
            elif red_ratio > self.nu_2:
                self.lr = min(self.max_lr, self.inc_factor * self.lr)
                break

            if stop:
                break

            if red_ratio < self.nu:
                scale = (
                    (-self.lr / old_lr * scale)
                    if c == 0
                    else (self.lr / old_lr * scale)
                )
            else:
                break

            self._apply_update(grad, scale)
            new_loss = closure(compute_grad=False)
            c += 1

        return new_loss
