import torch
import torch.distributed as dist
from .base_pmw_model import BasePMWModel


class WeightParallelizedTensor(BasePMWModel):
    def __init__(self, tensor, backend, master_group, rank):
        super().__init__()
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank
        self.device = self.default_device

    def detach(self):
        return torch.cat([p.detach().view(-1) for p in self.tensor])

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        new_args = tuple(x.detach() if isinstance(x, cls) else x for x in args)
        new_kwargs = {k: (v.detach() if isinstance(v, cls) else v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    def norm(self, p=2):
        if p in [2, float("inf")]:
            return torch.norm(self.detach(), p=p).item()
        raise NotImplementedError("Only L2 and L-inf norms are implemented.")

    def numel(self):
        return sum(p.numel() for p in self.tensor)

    def clone(self):
        return WeightParallelizedTensor(
            [p.clone() for p in self.tensor],
            backend=self.backend,
            master_group=self.master_group,
            rank=self.rank,
        )

    def __iter__(self):
        return iter(self.tensor)

    def __repr__(self):
        return f"Rank {self.rank}\nTensor: {self.tensor}"

    def __neg__(self):
        return WeightParallelizedTensor(
            [-p for p in self.tensor],
            backend=self.backend,
            master_group=self.master_group,
            rank=self.rank,
        )

    def __mul__(self, other):
        if isinstance(other, WeightParallelizedTensor):
            dp = torch.dot(self.detach(), other.detach())
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if self.backend != "gloo" else torch.device("cpu")
            dp = dp.to(device)
            dist.all_reduce(dp, group=self.master_group, op=dist.ReduceOp.SUM)
            return dp.item()
        elif isinstance(other, (int, float, torch.Tensor)):
            return WeightParallelizedTensor(
                [p * other for p in self.tensor],
                backend=self.backend,
                master_group=self.master_group,
                rank=self.rank,
            )
        return NotImplemented

    __matmul__ = __mul__
    __rmatmul__ = __mul__
    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, WeightParallelizedTensor):
            return WeightParallelizedTensor(
                [p + q for p, q in zip(self.tensor, other.tensor)],
                backend=self.backend,
                master_group=self.master_group,
                rank=self.rank,
            )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, WeightParallelizedTensor):
            return WeightParallelizedTensor(
                [p - q for p, q in zip(self.tensor, other.tensor)],
                backend=self.backend,
                master_group=self.master_group,
                rank=self.rank,
            )
        return NotImplemented
