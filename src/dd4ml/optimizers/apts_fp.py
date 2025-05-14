import torch, copy, random, inspect
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.optim as optim  # Optimizers
import numpy as np
from optimizers.TR import *

import torch, copy, random, inspect
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.optim as optim  # Optimizers
import numpy as np
from optimizers.TR import *


class APTS_W(Optimizer):
    def __name__(self):
        return "APTS_W"

    def __init__(
        self,
        params,
        model=None,
        loss_fn=None,
        device=None,
        max_iter=2,
        nr_models=None,
        global_opt=None,
        global_opt_params=None,
        local_opt=None,
        local_opt_params=None,
        global_pass=True,
        R=None,
        forced_decreasing_loss=False,
    ):

        super(APTS_W, self).__init__(params, {})

        self.device = (
            device
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.model = model
        self.max_iter = max_iter
        if nr_models is None:
            nr_models = len(list(self.model.parameters()))
        self.nr_models = nr_models
        self.global_pass = bool(global_pass)  # in case global_pass is 0 or 1
        self.TRargs = list(inspect.signature(TR).parameters.keys())
        self.loss_fn = loss_fn
        self.iter = 0
        self.forced_decreasing_loss = bool(
            forced_decreasing_loss
        )  # in case forced_decreasing_loss is 0 or 1

        # TODO: The current way to compute the layers is not correct. Many layers are made by two vectors (e.g. bias and weight)
        tot_layers = len(list(self.model.parameters()))
        self.tot_params = sum([p.numel() for p in self.model.parameters()])
        if tot_layers < self.nr_models:
            self.nr_models = tot_layers

        index_shuffled = torch.randperm(tot_layers)
        self.trainable_layers = []
        params_per_subset = [0] * self.nr_models
        # self.trainable_params = [0]*self.nr_models
        for i in range(self.nr_models):
            params_per_subset[i] = int(tot_layers / (nr_models - i))
            if i == self.nr_models - 1:
                self.trainable_layers.append(
                    index_shuffled[sum(params_per_subset[:i]) :]
                )
            else:
                self.trainable_layers.append(
                    index_shuffled[
                        sum(params_per_subset[:i]) : sum(params_per_subset[: i + 1])
                    ]
                )
            # for j in self.trainable_layers[i]:
            # self.trainable_params[i] += list(self.model.parameters())[j].numel()
            tot_layers -= params_per_subset[i]

        if "SGD" not in str(
            local_opt
        ):  # local opt options are optim.SGD and whatever else
            local_opt_params["min_radius"] = (
                0  # avoids problems with "radius" being smallers than "min_radius"
            )
            local_opt_params.update({"device": self.device})
        self.l_o = lambda p: local_opt(params=p, **local_opt_params)

        # Initialize the local optimizers
        self.SubModels = []
        self.streams = []
        self.Local_Optimizer = []
        for i in range(self.nr_models):
            self.streams.append(torch.cuda.Stream(device=self.device))
            self.SubModels.append(self.create_sub_model(self.trainable_layers[i]))
            self.Local_Optimizer.append(self.l_o(self.SubModels[i].parameters()))

        self.global_optimizer = global_opt(self.model.parameters(), **global_opt_params)
        self.radius = self.global_optimizer.radius

        self.R = []
        if R is None:
            for k in range(self.nr_models):
                # R[k] is a sparse matrix with "tot_params" columns and "self.trainable_params[k]" rows.
                # With 1 in the position of the parameters that are updated by the k-th local optimizer.
                # There is no need to build R explicitly, it is enough to restrict the gradient and the parameters according to the trainable layers of the model.
                self.R.append(None)
        else:
            raise ValueError("R != None not fully implemented yet")
            for k in range(self.nr_models):
                self.R.append(R[k])

        def l_closure(inputs, targets, model, k=[]):
            # self.SubModels[k].zero_grad()
            # get params of self.SubModels[k]
            # params = [p for p in self.SubModels[k].parameters() if p.requires_grad]
            J = self.loss_fn(model(inputs), targets)
            if torch.is_grad_enabled():
                # print(f'NN is differentiable: {self._is_differentiable(self.SubModels[k])}')
                J.backward()
            # print(f' this should be 0 : {(self.restricted_global_grad(k) - self.submodel_grad(k)) @ (self.submodel_params(k) - self.restricted_global_params(k))}')
            # TODO: Write correctly the following line... note that gradients should be the ones at the begin of the first iteration of the local optimizer
            return (
                J.detach()
            )  # + (self.restricted_global_grad(k) - self.submodel_grad(k)) @ (self.submodel_params(k) - self.restricted_global_params(k))

        def g_closure(inputs, targets):
            J = self.loss_fn(self.model(inputs), targets)
            if torch.is_grad_enabled():
                J.backward()
            # return J.detach()
            return J

        # self.local_closure = lambda inputs, targets, model, k: l_closure(inputs, targets, model, k)
        self.global_closure = lambda inputs, targets: g_closure(inputs, targets)
        self.old_loss = None

    def local_closure(self, inputs, targets, k):
        J = self.loss_fn(self.SubModels[k](inputs), targets)
        if torch.is_grad_enabled():
            self.SubModels[k].zero_grad()
            J.backward()
        # return J.detach()
        return J

    def restricted_global_grad(self, k):
        if self.R[k] is None:
            sub = list(self.SubModels[k].parameters())
            return torch.cat(
                [
                    p.grad.flatten().detach()
                    for i, p in enumerate(self.model.parameters())
                    if sub[i].grad is not None
                ]
            )
        else:
            return self.R[k] @ torch.cat(
                [p.grad.flatten().detach() for p in self.model.parameters()]
            )

    def restricted_global_params(self, k):
        if self.R[k] is None:
            sub = list(self.SubModels[k].parameters())
            return torch.cat(
                [
                    p.flatten().detach()
                    for i, p in enumerate(self.model.parameters())
                    if sub[i].requires_grad is True
                ]
            )
        else:
            return self.R[k] @ torch.cat(
                [p.flatten().detach() for p in self.model.parameters()]
            )

    def submodel_params(self, k):
        if self.R[k] is None:
            return torch.cat(
                [
                    p.flatten().detach()
                    for p in self.SubModels[k].parameters()
                    if p.requires_grad is True
                ]
            )
        else:
            pass

    def submodel_grad(self, k):
        if self.R[k] is None:
            return torch.cat(
                [
                    p.grad.flatten().detach()
                    for p in self.SubModels[k].parameters()
                    if p.grad is not None
                ]
            )
        else:
            pass

    def _params(self, model):
        # return torch.cat([p.flatten().detach() for p in model.parameters()])
        return torch.cat([p.flatten() for p in model.parameters()])

    def create_sub_model(self, trainable_layers, k=None):
        """
        Creates a submodel with the trainable layers specified by weight_subset
        NOTE: Updating the SubNetworks in place of deepcopy seems to yield plenty of errors
        NOTE 2: if all layers are learnable, for some reason everything works fine
        """
        with torch.no_grad():
            if k is None:
                local_model = copy.deepcopy(self.model)
                for index, param in enumerate(local_model.parameters()):
                    if index in trainable_layers:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                return local_model
            else:
                self.SubModels[k].load_state_dict(self.model.state_dict())

    def local_net_sync(self):
        # TODO: speed this up by using streams or multiprocessing?
        for i in range(self.nr_models):
            # sync the nets
            self.create_sub_model(self.trainable_layers[i], k=i)
            # sync the optimizers
            for key in self.global_optimizer.collection_of_all_variables:
                if key == "hessian_approx":
                    setattr(
                        self.Local_Optimizer[i],
                        key,
                        copy.deepcopy(getattr(self.global_optimizer, key)),
                    )
                    if self.Local_Optimizer[i].hessian_approx.S is not None:
                        v1 = []
                        v2 = []
                        v3 = []
                        a = 0
                        a2 = 0
                        for j, p in enumerate(self.model.parameters()):
                            b2 = a2 + p.numel()
                            if j in self.trainable_layers[i]:
                                b = a + p.numel()
                                v1.append(
                                    self.Local_Optimizer[i].hessian_approx.S[a2:b2]
                                )
                                v2.append(
                                    self.Local_Optimizer[i].hessian_approx.Y[a2:b2]
                                )
                                v3.append(
                                    self.Local_Optimizer[i].hessian_approx.Psi[a2:b2]
                                )
                                a = b
                            a2 = b2
                        try:
                            self.Local_Optimizer[i].hessian_approx.S = (
                                torch.concatenate(v1)
                            )
                            self.Local_Optimizer[i].hessian_approx.Y = (
                                torch.concatenate(v2)
                            )
                            self.Local_Optimizer[i].hessian_approx.Psi = (
                                torch.concatenate(v3)
                            )
                        except:
                            self.Local_Optimizer[i].hessian_approx.S = np.concatenate(
                                v1
                            )
                            self.Local_Optimizer[i].hessian_approx.Y = np.concatenate(
                                v2
                            )
                            self.Local_Optimizer[i].hessian_approx.Psi = np.concatenate(
                                v3
                            )
                elif key in ["var", "mom"]:
                    v = []
                    v2 = getattr(self.global_optimizer, key)
                    a = 0
                    a2 = 0
                    for j, p in enumerate(self.model.parameters()):
                        b2 = a2 + p.numel()
                        if j in self.trainable_layers[i]:
                            b = a + p.numel()
                            v.append(v2[a2:b2])
                            a = b
                        a2 = b2
                    setattr(self.Local_Optimizer[i], key, torch.cat(v))
                else:
                    setattr(
                        self.Local_Optimizer[i],
                        key,
                        getattr(self.global_optimizer, key),
                    )

    def local_steps(self, local_closure):
        # update the local model
        TEMP = 2
        local_loss = [0 for _ in range(self.nr_models)]
        lr = [self.radius / TEMP] * self.nr_models
        g = torch.zeros(self.tot_params, device=self.device)
        grad_norm2 = 0
        # start_time = time.time()
        for i in range(self.nr_models):
            # with torch.cuda.stream(self.streams[i]):
            for j in range(self.max_iter):  # Iterations of the preconditioning step
                if j == 0:
                    self.Local_Optimizer[i].radius = lr[i]
                else:
                    self.Local_Optimizer[i].radius = lr[i]
                self.Local_Optimizer[i].max_radius = lr[i]
                # print(f'Iteration {j}: (PRE) max radius {self.Local_Optimizer[i].max_radius}, radius {self.Local_Optimizer[i].radius}')

                try:
                    assert lr[i].isnan() == False
                except:
                    assert np.isnan(lr[i]) == False

                assert self.submodel_params(i).isnan().sum() == 0

                local_loss[i], grad, grad_norm = self.Local_Optimizer[i].step(
                    lambda x=1: local_closure(i)
                )

                assert self.submodel_params(i).isnan().sum() == 0

                if j == 0:
                    grad_norm2 = max(grad_norm2, grad_norm)
                    a = 0
                    a2 = 0
                    for p in self.SubModels[i].parameters():
                        b2 = a2 + p.numel()
                        if p.requires_grad:
                            b = a + p.numel()
                            assert g[a2:b2].abs().sum() == 0
                            assert grad[a:b].abs().sum() != 0
                            g[a2:b2] = grad[a:b]
                            a = b
                        a2 = b2

                # Update radius to make sure the update is inside the global trust region radius
                lr[i] = self.radius / TEMP - torch.norm(
                    self.restricted_global_params(i) - self.submodel_params(i),
                    p=torch.inf,
                )

                try:
                    assert lr[i].isnan() == False
                except:
                    assert np.isnan(lr[i]) == False

                # print(f'Iteration {j}: (POST) radius {lr[i]}, TR radius {self.Local_Optimizer[i].radius}')
                if lr[i] < self.global_optimizer.min_radius / 10:
                    break
            # Final check
            if torch.norm(
                self.restricted_global_params(i) - self.submodel_params(i), p=torch.inf
            ) > self.radius / TEMP + 10**-3 or lr[i] < -(10**-3):
                print(
                    f"Step too large or negative radius: {torch.norm(self.restricted_global_params(i) - self.submodel_params(i), p=torch.inf)}>{self.radius/TEMP},   {lr[i]}"
                )

        # torch.cuda.synchronize(self.streams[i])
        # print(f'Local steps took {time.time()-start_time} seconds')
        return local_loss, g, grad_norm2

    def get_local_params(self):
        """
        Get all local trainable parameters
        """
        with torch.no_grad():
            local_params = [0] * len(list(self.model.parameters()))
            for i in range(self.nr_models):
                for j, p in enumerate(self.Local_Optimizer[i].params):
                    if j in self.trainable_layers[i]:
                        local_params[j] = p.data
        return local_params  # torch.cat(local_params)

    def global_params_update(self, local_params):
        """
        Update the global model with the average of the local models
        """
        with torch.no_grad():
            # if local_params is a list
            if isinstance(local_params, list):
                for j, p in enumerate(self.model.parameters()):
                    p.data.copy_(local_params[j])
            else:
                a = 0
                for j, p in enumerate(self.model.parameters()):
                    # take the correct amount of parameters from local_params and reshape them into the shape of p
                    p.data.copy_(local_params[a : a + p.numel()].reshape(p.shape))
                    a += p.numel()

    def step(self, inputs, targets):
        """
        One step of APTS
        """
        self.iter += 1
        skip_preconditioning = False
        if skip_preconditioning is False:
            # check current loss of local models:
            # with torch.no_grad():
            # initial_loss = [0 for _ in range(self.nr_models)]
            # for i in range(self.nr_models):
            #     initial_loss[i] = self.local_closure(inputs,targets,i)
            #     initial_loss[i] = round(initial_loss[i].item(),3)

            self.local_net_sync()  # Sharing weights of the global NN with the local ones
            a = time.time()
            local_loss, g, g_norm = self.local_steps(
                lambda k: self.local_closure(inputs, targets, k)
            )
            print(f"Local steps took {time.time()-a} seconds")

            for i in range(len(local_loss)):  ### erease this
                local_loss[i] = round(local_loss[i], 3)  ###

            # check that the loss is decreasing
            if self.forced_decreasing_loss:
                x_loc = torch.cat([p.flatten() for p in self.get_local_params()])
                x_glob = self._params(self.model)
                step = x_loc - x_glob
                step = step / torch.norm(step, p=torch.inf)
                g = g / torch.norm(g, p=torch.inf)
                if self.old_loss is None:  # old loss
                    with torch.no_grad():
                        self.old_loss = self.global_closure(inputs, targets)
                self.global_params_update(
                    self.get_local_params()
                )  # update global model
                with torch.no_grad():
                    loss = self.global_closure(inputs, targets)  # current loss
                radius = self.radius
                w = 0
                c = 0
                while loss > self.old_loss:
                    c += 1
                    # print(f'----------> Loss is increasing ({loss} > {self.old_loss}), reducing step size. W = '+ str(w))
                    radius = radius / 2
                    w = min(w + 0.2, 1)
                    step2 = step * (1 - w) + w * g
                    x_loc = x_glob + step2 / torch.norm(step2, p=torch.inf) * radius
                    self.global_params_update(x_loc)
                    with torch.no_grad():
                        loss = self.global_closure(inputs, targets)
                    if c > 5:
                        print(
                            f"Warning: APTS is not decreasing the loss {loss} > {self.old_loss}"
                        )
                        break
            else:
                self.global_params_update(self.get_local_params())

        # Global smoothing step
        if self.global_pass:
            new_loss = self.global_optimizer.step(
                lambda x=1: self.global_closure(inputs, targets)
            )[0]
            self.radius = self.global_optimizer.radius

        self.old_loss = new_loss
        # print(f'Iteration {self.iter} | initial loss: {initial_loss}, after local steps: {local_loss}, loss before global step {loss}, final loss {new_loss}') ##
        if self.iter == 1:
            if skip_preconditioning:
                print("Skipping preconditioning")
            else:
                if self.forced_decreasing_loss:
                    print("Modified APTS with forced decreasing loss")

        return new_loss, 2  # dummy output
