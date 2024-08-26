import torch
import torch.distributed as dist
from pmw.base_model import BaseModel


class Something(BaseModel):
    def __init__(self):
        super().__init__()

s = Something()
# Print s's parameter norm
print(f'Rank {dist.get_rank()} param norm: {torch.norm(torch.cat([p.flatten() for p in s.parameters()]))}')