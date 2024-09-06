import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._tensor import Shard, distribute_tensor, init_device_mesh
import time 

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def main(rank=None, master_addr=None, master_port=None, world_size=None):
    # Suppose we have 4 ranks in total, i.e. ranks 0, 1, 2, and 3
    prepare_distributed_environment(rank, master_addr, master_port, world_size)
    # Shard first tensor across ranks [0,1], multliply it with a weight matrix, and send the output to ranks [2,3]
    rowwise_placement=[Shard(0)]
    our_ranks = [0,1]
    next_ranks = [2,3]
    dist.barrier()
    time.sleep(rank)
    device_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dim0", "dim1")) 
    # dummy sharded tensor
    matrix = torch.eye(10, 10)
    sharded_matrix = distribute_tensor(matrix, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
    if rank in [0,1]:
        # Define some tensors for testing
        vector = torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.float32).view(10, 1)
        matrix = torch.eye(10, 10) 

        # NOTE: Currently the code gets stuck HERE because these functions require every rank to call them
        dm = device_mesh["dim1"]
        sharded_matrix = distribute_tensor(matrix, device_mesh=dm, placements=rowwise_placement)
        sharded_vector = distribute_tensor(vector, device_mesh=dm, placements=rowwise_placement)
        
        for _ in range(10):
            output = sharded_matrix @ sharded_vector
        output = output._local_tensor.cpu()
        
        v = [0,1,2,3,4] if rank == 0 else [5,6,7,8,9]
        solution = torch.tensor(v, dtype=torch.float32).view(5, 1)
        print(f"Rank {rank} sending output to {next_ranks[our_ranks.index(rank)]} - error: {torch.norm(output - solution)}")
        dist.send(tensor=output, dst=next_ranks[our_ranks.index(rank)])
        print(f'Rank {rank} sent output to {next_ranks[our_ranks.index(rank)]}')
        
    if rank in [2,3]:
        vector = torch.ones(10, 1)
        sharded_input = distribute_tensor(vector, device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        sharded_matrix = distribute_tensor(2*torch.eye(10, 10), device_mesh=device_mesh["dim1"], placements=rowwise_placement)
        # We update the local tensor with the received tensor
        temp = sharded_input._local_tensor.cpu()
        print(f"Rank {rank} receiving tensor from {our_ranks[next_ranks.index(rank)]}")
        dist.recv(tensor=temp, src=our_ranks[next_ranks.index(rank)])
        print(f'Rank {rank} received tensor from {our_ranks[next_ranks.index(rank)]}')
        sharded_input._local_tensor = temp.to('cuda')
        
        # product:
        output = sharded_matrix @ sharded_input
        
        v = [0,2,4,6,8] if rank == 2 else [10,12,14,16,18]
        solution = torch.tensor(v, dtype=torch.float32).view(5, 1)
        print(f"Rank {rank} received tensor - error {torch.norm(output._local_tensor.cpu() - solution)}")
        
        # At this point we want to perform computations with the received tensor and another sharded matrix by only using ranks [2,3]
    
    print(f"Rank {rank} asd")

if __name__ == '__main__':  
    world_size = 4
    mp.spawn(main, args=('localhost', '12345', world_size), nprocs=world_size, join=True)