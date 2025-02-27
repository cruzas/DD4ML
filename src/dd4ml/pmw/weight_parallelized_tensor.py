import math
import torch
import torch.distributed as dist
from dd4ml.pmw.base_pmw_model import BasePMWModel

class WeightParallelizedTensor(BasePMWModel):
    def __init__(self, tensor, backend, master_group, rank):
        super().__init__()
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank
        self.device = self.default_device

    def detach(self):
        # Return a plain flattened tensor from the underlying list.
        return torch.cat([p.detach().view(-1) for p in self.tensor])

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Replace any WeightParallelizedTensor with its detached plain tensor.
        new_args = tuple(x.detach() if isinstance(x, WeightParallelizedTensor) else x for x in args)
        new_kwargs = {k: (v.detach() if isinstance(v, WeightParallelizedTensor) else v)
                      for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    def norm(self, p=2):
        if p == 2:
            flat = self.detach()
            return torch.norm(flat, p=2).item()
        elif p == float("inf"):
            flat = self.detach()
            return torch.norm(flat, p=float("inf")).item()
        else:
            raise NotImplementedError("Only L2 and L-inf norms are implemented.")

    def numel(self):
        return sum(p.numel() for p in self.tensor)

    def clone(self):
        return WeightParallelizedTensor([p.clone() for p in self.tensor],
                                        backend=self.backend,
                                        master_group=self.master_group,
                                        rank=self.rank)

    def __iter__(self):
        return iter(self.tensor)

    def __repr__(self):
        return f'Rank {self.rank}\nTensor: {self.tensor}'

    def __neg__(self):
        return WeightParallelizedTensor([-p for p in self.tensor],
                                        backend=self.backend,
                                        master_group=self.master_group,
                                        rank=self.rank)

    def __matmul__(self, other):
        return self.__mul__(other)

    def __rmatmul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, a):
        return self.__mul__(a)

    def __mul__(self, other):
        if isinstance(other, WeightParallelizedTensor):
            g1 = self.detach()
            g2 = other.detach()
            dot_product = torch.dot(g1, g2)
            device = torch.device(f'cuda:{torch.cuda.current_device()}') if self.backend != 'gloo' else torch.device('cpu')
            dot_product = dot_product.to(device)
            dist.all_reduce(dot_product, group=self.master_group, op=dist.ReduceOp.SUM)
            return dot_product.item()
        elif isinstance(other, (int, float, torch.Tensor)):
            new_tensor = [p * other for p in self.tensor]
            return WeightParallelizedTensor(new_tensor, backend=self.backend,
                                            master_group=self.master_group, rank=self.rank)
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, WeightParallelizedTensor):
            return WeightParallelizedTensor([p + q for p, q in zip(self.tensor, other.tensor)],
                                            backend=self.backend,
                                            master_group=self.master_group,
                                            rank=self.rank)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, WeightParallelizedTensor):
            return WeightParallelizedTensor([p - q for p, q in zip(self.tensor, other.tensor)],
                                            backend=self.backend,
                                            master_group=self.master_group,
                                            rank=self.rank)
        return NotImplemented