import math

import torch
import torch.distributed as dist

from src.pmw.base_model import BaseModel


class WeightParallelizedTensor(BaseModel):
    def __init__(self, tensor, backend, master_group, rank):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank

    def norm(self, p=2):
        if p == 2:
            return math.sqrt(self @ self)
        elif p == torch.tensor(float('inf')):
            local_max = torch.tensor(
                max([p.flatten().abs().max().item() for p in self.tensor]))
            dist.all_reduce(tensor=local_max,
                            group=self.master_group, op=dist.ReduceOp.MAX)
            return local_max.item()
        else:
            # Implement generic p
            raise NotImplementedError("Only L2 norm is implemented.")

    def __iter__(self):
        return iter(self.tensor)

    def __repr__(self):
        return f'Rank {self.rank}\nGradient: {self.model.subdomain.grad()}'

    def __matmul__(self, a):  # self.grad @ a
        return self * a

    def __rmatmul__(self, a):  # a @ self.grad
        return a * self

    def __rmul__(self, a):  # a * self.grad
        # This handles the commutative property of multiplication
        return self.__mul__(a)

    def __mul__(self, a):  # self.grad * a
        # When both operands are WeightParallelizedTensor instances
        if isinstance(a, WeightParallelizedTensor):
            g1 = torch.cat([p.flatten() for p in self.tensor],
                           dim=0)  # Flatten the gradients
            g2 = torch.cat([p.flatten() for p in a.tensor],
                           dim=0)  # Flatten the gradients
            g3 = g1 @ g2
            g3 = g3.to(f'cpu' if self.backend == 'gloo' else f'cuda:0')
            # Sum the gradients on the master rank
            dist.all_reduce(tensor=g3, group=self.master_group,
                            op=dist.ReduceOp.SUM)
            return g3.item()
        else:
            # Multiply model by a scalar or tensor
            return WeightParallelizedTensor([p*a for p in self.tensor], backend=self.backend, master_group=self.master_group, rank=self.rank)

    def __add__(self, a):
        if isinstance(a, WeightParallelizedTensor):
            return WeightParallelizedTensor([p+q for p, q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)

    def __sub__(self, a):
        if isinstance(a, WeightParallelizedTensor):
            return WeightParallelizedTensor([p-q for p, q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)
