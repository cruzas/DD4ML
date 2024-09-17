import torch
from functools import reduce
from torch.optim import Optimizer
from trainers.Lineasearches import _cubic_interpolate, _strong_wolfe
import torch.distributed as dist

__all__ = ['GDLinesearch']


class GDLinesearch(Optimizer):

    def __init__(self,
                 params,
                 max_iter=1.0,
                 line_search_fn="strong_wolfe"):
        # if max_eval is None:
        #     max_eval = max_iter * 5 // 4
        defaults = dict(max_iter=max_iter, line_search_fn=line_search_fn)
        super(GDLinesearch, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("GDLinesearch doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.num_fun_evals = 0

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat(self, v):
        views = []
        for p in v:
            # if p.grad is None:
            #     view = p.new(p.numel()).zero_()
            # el
            if p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)        
        

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad


    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # print("Taking step ...... ")

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        max_iter = group['max_iter']
        line_search_fn = group['line_search_fn']

        state = self.state[self._params[0]]
        state.setdefault('n_iter', 0)


        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        # state['func_evals'] += 1


        flat_grad = self._gather_flat_grad()


        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')


        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            d = flat_grad.neg()

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            t = 1.0
            gtd = flat_grad.dot(d)  # g * d


            # perform line search
            x_init = self._clone_param()

            def obj_func(x, t, d):
                return self._directional_evaluate(closure, x, t, d)

            # it would be good to get rid of flat, but line-search works better as it is 
            loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                obj_func, x_init, t, d, loss, flat_grad, gtd)


            self.num_fun_evals += ls_func_evals

            # if(dist.get_rank()==0):
            #     print("--- ls ", t, "loss loc  ", loss)


            self._add_grad(t, d)




            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break



        state['d'] = d
        state['t'] = t


        return orig_loss
