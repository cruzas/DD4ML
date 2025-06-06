import torch
import torch.distributed as dist
from .base_pmw_model import BasePMWModel
from dd4ml.utility.ml_utils import get_device


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

        # Intercept torch.norm
        if func is torch.norm and isinstance(args[0], cls):
            return args[0].norm(**kwargs)
        elif func is torch.zeros_like and isinstance(args[0], cls):
            like = args[0]
            zeros = [torch.zeros_like(t) for t in like.tensor]
            return cls(zeros, like.backend, like.master_group, like.rank)

        # Fallback: detach any WPTs and delegate to PyTorch
        new_args = tuple(x.detach() if isinstance(x, cls) else x for x in args)
        new_kwargs = {k: (v.detach() if isinstance(v, cls) else v)
                      for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    def norm(self, p=2, dim=None, keepdim=False, **ignored):
        """
        Compute the vector-norm across all shards.

        Only global (``dim is None``) L-2 and L-∞ are implemented.
        Any extra kwargs coming from ``torch.norm`` are silently ignored.
        """
        if dim is not None:
            raise NotImplementedError("Per-dimension norms are not supported yet.")
        if p in (2, float("inf")):
            return torch.norm(self.detach(), p=p)
        raise NotImplementedError("Only L-2 and L-∞ norms are implemented.")

    def shape(self):
        """
        Return the full tensor shape across all ranks.

        Assumes shards are split along dimension 0, and that each shard
        on every rank shares identical trailing dimensions.
        """
        # Compute local size along dim 0 (sum over all local shards)
        local_dim0 = sum(p.size(0) for p in self.tensor)
        dim0_tensor = torch.tensor(local_dim0, device=get_device())

        # Gather all local_dim0 values from every rank
        world_size = dist.get_world_size(self.master_group)
        gathered = [torch.zeros_like(dim0_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, dim0_tensor, group=self.master_group)

        # Sum to get global size along dim 0
        global_dim0 = sum(int(x.item()) for x in gathered)

        # The remaining dimensions are the same on all shards
        rest = list(self.tensor[0].shape[1:])
        return tuple([global_dim0] + rest)

    def dim(self):
        """
        Return the number of dimensions of the full tensor.
        """
        return len(self.shape())

    def numel(self):
        return sum(p.numel() for p in self.tensor)

    def clone(self):
        return WeightParallelizedTensor(
            [p.clone() for p in self.tensor],
            backend=self.backend,
            master_group=self.master_group,
            rank=self.rank,
        )

    def to_device(self, device):
        """
        Move all local shards to the specified device.
        """
        self.tensor = [p.to(device) for p in self.tensor]
        self.device = device
        return self

    def __iter__(self):
        return iter(self.tensor)

    def __repr__(self):
        return f"Rank {self.rank}\nTensor shards: {self.tensor}"

    def __neg__(self):
        return WeightParallelizedTensor(
            [-p for p in self.tensor],
            backend=self.backend,
            master_group=self.master_group,
            rank=self.rank,
        )

    def __mul__(self, other):
        """
        If `other` is another WeightParallelizedTensor, perform a distributed dot-product.
        If `other` is scalar or torch.Tensor, scale each local shard by `other`.
        """
        if isinstance(other, WeightParallelizedTensor):
            local_dp = torch.dot(self.detach(), other.detach()).to(get_device())
            dist.all_reduce(local_dp, group=self.master_group, op=dist.ReduceOp.SUM)
            return local_dp

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

    def dot(self, other):
        """
        If `other` is a WeightParallelizedTensor, perform a distributed dot-product.
        If `other` is a scalar (int/float) or torch.Tensor, scale each local shard.
        """
        if isinstance(other, WeightParallelizedTensor):
            local_dp = torch.dot(self.detach(), other.detach()).to(get_device())
            dist.all_reduce(local_dp, group=self.master_group, op=dist.ReduceOp.SUM)
            return local_dp

        elif isinstance(other, (int, float, torch.Tensor)):
            return WeightParallelizedTensor(
                [p * other for p in self.tensor],
                backend=self.backend,
                master_group=self.master_group,
                rank=self.rank,
            )

        return NotImplemented

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

    def __iadd__(self, other):
        """
        In-place addition of another WeightParallelizedTensor.
        """
        if not isinstance(other, WeightParallelizedTensor):
            return NotImplemented
        for p, q in zip(self.tensor, other.tensor):
            p.add_(q)
        return self

    def __isub__(self, other):
        """
        In-place subtraction of another WeightParallelizedTensor.
        """
        if not isinstance(other, WeightParallelizedTensor):
            return NotImplemented
        for p, q in zip(self.tensor, other.tensor):
            p.sub_(q)
        return self

    def __imul__(self, other):
        """
        In-place scaling by scalar or torch.Tensor.
        """
        if isinstance(other, (int, float, torch.Tensor)):
            for p in self.tensor:
                p.mul_(other)
            return self
        return NotImplemented

    def div(self, scalar):
        """
        Divide each local shard by a scalar value (elementwise).
        """
        if not isinstance(scalar, (int, float, torch.Tensor)):
            raise TypeError("div: scalar must be int, float, or torch.Tensor")
        return WeightParallelizedTensor(
            [p / scalar for p in self.tensor],
            backend=self.backend,
            master_group=self.master_group,
            rank=self.rank,
        )

    def sum(self):
        """
        Sum all elements across local shards, then reduce globally.
        """
        local_sum = sum(p.sum() for p in self.tensor).to(get_device())
        dist.all_reduce(local_sum, group=self.master_group, op=dist.ReduceOp.SUM)
        return local_sum

    def mul_(self, other):
        """
        In-place element-wise multiplication.

        * If ``other`` is another WeightParallelizedTensor, multiply shard-wise.
        * If ``other`` is a scalar or ``torch.Tensor``, broadcast-multiply every shard.
        """
        if isinstance(other, WeightParallelizedTensor):
            for p, q in zip(self.tensor, other.tensor):
                p.mul_(q)
            return self

        if isinstance(other, (int, float, torch.Tensor)):
            for p in self.tensor:
                p.mul_(other)
            return self

        return NotImplemented

    def add_(self, other, alpha: float = 1.0):
        """
        In-place element-wise addition.

        * If ``other`` is another WeightParallelizedTensor, add shard-wise.
        * If ``other`` is a scalar or ``torch.Tensor``, broadcast-add every shard.

        The ``alpha`` parameter matches ``torch.Tensor.add_``.
        """
        if isinstance(other, WeightParallelizedTensor):
            for p, q in zip(self.tensor, other.tensor):
                p.add_(q, alpha=alpha)
            return self

        if isinstance(other, (int, float, torch.Tensor)):
            for p in self.tensor:
                p.add_(other, alpha=alpha)
            return self

        return NotImplemented
    
    @property
    def dtype(self):
        return self.tensor[0].dtype if self.tensor else torch.float32
    
    def clone(self):
        """
        Create a deep copy of this WeightParallelizedTensor, preserving:
        - each shard's values, dtype, and device
        - the backend, master_group, and rank metadata
        """
        # Clone each local shard (shard.clone() preserves dtype & device)
        cloned_shards = [p.clone() for p in self.tensor]

        # Construct a new WPT with the same metadata
        new_wpt = WeightParallelizedTensor(
            cloned_shards,
            backend=self.backend,
            master_group=self.master_group,
            rank=self.rank,
        )

        # If you'd rather store device as an explicit attribute instead of using property(),
        #    you can set new_wpt._device here; however, since `device` is now a @property
        #    that inspects each shard, no further action is required. :contentReference[oaicite:0]{index=0}

        new_wpt.device = self.device  # Ensure the device is set correctly
        return new_wpt
