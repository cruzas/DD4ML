import torch
from functools import reduce
from torch.optim import Optimizer
from trainers.Lineasearches import _cubic_interpolate, _strong_wolfe

__all__ = ['PrecLBFGS']


class PrecLBFGS(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.
    """
    def __init__(self,
                 params,
                 history_size=10):
        defaults = dict(
            history_size=history_size)
        super(PrecLBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("PrecLBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params            = self.param_groups[0]['params']
        self._numel_cache       = None

        self._momentum_param    = 0.9  # eqvivalent to no momentum 1.0, 0.9
        self.num_fun_evals      = 0


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


    def _gather_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return views      


    def _gather_flat(self, v):
        views = []
        for p in v:
            if p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)        


    def _add_dir(self, step_size, update):
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
        self._add_dir(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        
        self._set_param(x)
        return loss, flat_grad


    @torch.no_grad()
    def step(self, closure):

        assert len(self.param_groups) == 1

        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        history_size = group['history_size']


        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)


        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)

        flat_grad = self._gather_flat_grad()


        # tensors cached in state (for tracing)
        momentum = state.get('momentum')

        d = state.get('d')
        
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')

        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')


        n_iter = 0

        # keep track of nb of iterations
        n_iter += 1
        state['n_iter'] += 1


        ############################################################
        # compute gradient descent direction
        ############################################################
        if state['n_iter'] == 1:
            d = flat_grad.neg()
            old_dirs = []
            old_stps = []
            ro = []
            H_diag = 1
            momentum = d.clone()
            momentum = 0.0*momentum

        else:
            # do lbfgs update (update memory)
            y = flat_grad.sub(prev_flat_grad)
            yd = y.dot(d)  


            if yd > 1e-15:
                # updating memory
                if len(old_dirs) == history_size:
                    # shift history by one (limited-memory)
                    old_dirs.pop(0)
                    old_stps.pop(0)
                    ro.pop(0)

                # store new direction/step
                old_dirs.append(y)
                old_stps.append(d)
                ro.append(1./yd)

                # update scale of initial Hessian approximation
                H_diag = yd/y.dot(y)  # (y*y)


            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            num_old = len(old_dirs)


            if 'al' not in state:
                state['al'] = [None] * history_size
            al = state['al']


            # iteration in L-BFGS loop collapsed to use just one buffer
            q = flat_grad.neg()
            for i in range(num_old - 1, -1, -1):
                al[i] = old_stps[i].dot(q) * ro[i]
                q.add_(old_dirs[i], alpha=-al[i])


            # multiply by initial Hessian
            # r/d is the final direction
            d = r = torch.mul(q, H_diag)
            for i in range(num_old):
                be_i = old_dirs[i].dot(r) * ro[i]
                r.add_(old_stps[i], alpha=al[i] - be_i)


        if prev_flat_grad is None:
            prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
        else:
            prev_flat_grad.copy_(flat_grad)



        ############################################################
        # compute step length
        ############################################################

        # directional derivative
        gtd = flat_grad.dot(d)  # g*d


        # optional line search: user function
        x_init = self._clone_param()


        momentum = momentum.mul(1.0-self._momentum_param)
        offset = 0
        for m in momentum:
            numel = m.numel()
            m.add_(d[offset:offset + numel].view_as(m), alpha=self._momentum_param)
            offset += numel


        def obj_func(x, t, d):
            return self._directional_evaluate(closure, x, t, momentum)


        loss, flat_grad, t, ls_func_evals = _strong_wolfe(
            obj_func, x_init, 1.0, momentum, loss, flat_grad, gtd)


        # print("--- lr - - -  ", t)
        self._add_dir(t, momentum)
        d = momentum.mul(t)


        self.num_fun_evals += ls_func_evals

        # summarize and store stuff 
        state['d']          = d
        state['momentum']   = momentum

        state['old_dirs']   = old_dirs
        state['old_stps']   = old_stps

        state['ro']             = ro
        state['H_diag']         = H_diag
        state['prev_flat_grad'] = prev_flat_grad


        return orig_loss






