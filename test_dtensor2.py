import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Suppose we have 4 ranks in total, i.e. ranks 0, 1, 2, and 3
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    # Shard first tensor across ranks [0,1], multliply it with a weight matrix, and send the output to ranks [2,3]
    rowwise_placement=[Shard(0)]

    # Define the device mesh
    if rank in [0,1]:
        device_mesh = init_device_mesh("cuda", (2,))  
    # to parallelize tensors [t0, t1]
    # t0 is sharded across ranks [0,1] and t1 is sharded across ranks [2,3]
    
    # Define some tensors for testing
    vector = torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.float32).view(10, 1)
    matrix = torch.eye(10, 10) 
    # time.sleep(rank)
    # print(f"Rank {rank} dim0: {device_mesh['dim0']} and dim1: {device_mesh['dim1']}")
    
    # This prints out:
    # Rank 0 should yield [0,1]. Gives DeviceMesh([0, 1])
    # Rank 1 should yield [0,1]. Gives DeviceMesh([0, 1])
    # Rank 2 should yield [2,3]. Gives DeviceMesh([2, 3])
    # Rank 3 should yield [2,3]. Gives DeviceMesh([2, 3])
    
    # NOTE: Currently the code gets stuck HERE because these functions require every rank to call them
    if rank in [0, 1]:
        sharded_matrix = distribute_tensor(matrix, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        sharded_vector = distribute_tensor(vector, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        
        output = sharded_matrix @ sharded_vector
        print(f"Rank {rank} output {output}")

    # dist.send(tensor=output, dst=[2,3].index(rank))
        
    vector = torch.empty(10, 1)
    sharded_input = distribute_tensor(vector, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
    # We update the local tensor with the received tensor
    dist.recv(tensor=sharded_input._local_tensor, src=[0,1][[2,3].index(rank)])
    
    # At this point we want to perform computations with the received tensor and another sharded matrix by only using ranks [2,3]

if __name__ == '__main__':  
    world_size = 4
    mp.spawn(main, args=('localhost', '12345', world_size), nprocs=world_size, join=True)