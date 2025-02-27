import math
import torch
import torch.distributed as dist
from dd4ml.pmw.base_pmw_model import BasePMWModel

class WeightParallelizedTensor(BasePMWModel):
    def __init__(self, tensor, backend, master_group, rank):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank

    def norm(self, p=2):
        if p == 2:
            dot = self @ self
            return math.sqrt(dot)
        elif p == float("inf"):
            local_max = max(p.abs().max().item() for p in self.tensor)
            local_max_tensor = torch.tensor(local_max)
            dist.all_reduce(local_max_tensor, group=self.master_group,
                            op=dist.ReduceOp.MAX)
            return local_max_tensor.item()
        else:
            raise NotImplementedError("Only L2 and L-inf norms are implemented.")

    def __iter__(self):
        return iter(self.tensor)

    def __repr__(self):
        return f'Rank {self.rank}\nGradient: {self.model.subdomain.grad()}'

    def __matmul__(self, other):
        return self.__mul__(other)

    def __rmatmul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, a):  # a * self.grad
        # This handles the commutative property of multiplication
        return self.__mul__(a)

    def __mul__(self, other):
        if isinstance(other, WeightParallelizedTensor):
            g1 = torch.cat([p.view(-1) for p in self.tensor])
            g2 = torch.cat([p.view(-1) for p in other.tensor])
            dot_product = g1 @ g2
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

    def __add__(self, a):
        if isinstance(a, WeightParallelizedTensor):
            return WeightParallelizedTensor([p+q for p, q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)
        return NotImplemented

    def __sub__(self, a):
        if isinstance(a, WeightParallelizedTensor):
            return WeightParallelizedTensor([p-q for p, q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)
        return NotImplemented