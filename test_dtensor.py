import torch
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

# construct a device mesh with available devices (multi-host or single host)
device_mesh = init_device_mesh("cuda", (torch.cuda.device_count(),))
# if we want to do row-wise sharding
rowwise_placement=[Shard(0)]
# if we want to do col-wise sharding
colwise_placement=[Shard(1)]

big_tensor = torch.randn(888, 12)
# distributed tensor returned will be sharded across the dimension specified in placements
rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)

# if we want to do replication across a certain device list
replica_placement = [Replicate()]
# distributed tensor will be replicated to all four GPUs.
replica_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=replica_placement)

# if we want to distributed a tensor with both replication and sharding
device_mesh = init_device_mesh("cuda", (2, 2))
# replicate across the first dimension of device mesh, then sharding on the second dimension of device mesh
spec=[Replicate(), Shard(0)]
partial_replica = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=spec)

# create a DistributedTensor that shards on dim 0, from a local torch.Tensor
local_tensor = torch.randn((8, 8), requires_grad=True)
rowwise_tensor = DTensor.from_local(local_tensor, device_mesh, rowwise_placement)

# reshard the current row-wise tensor to a colwise tensor or replicate tensor
colwise_tensor = rowwise_tensor.redistribute(device_mesh, colwise_placement)
replica_tensor = colwise_tensor.redistribute(device_mesh, replica_placement)