import os
import pickle
import socket
import subprocess
import sys
from contextlib import closing

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb


def worker(
    rank,
    world_size,
    main_func,       # <--- The function to run (i.e., your 'main')
    sweep_id,
    entity,
    project,
    args_dict,
    use_wandb
):
    """
    Generic worker that spawns the main function on a given rank.
    """
    if use_wandb:
        wandb.agent(
            sweep_id=sweep_id,
            function=lambda: main_func(
                rank=rank,
                world_size=world_size,
                args=args_dict,
                wandb_config=wandb.config,
                use_wandb=True
            ),
            entity=entity,
            project=project,
            count=1
        )
    else:
        main_func(
            rank=rank,
            world_size=world_size,
            args=args_dict,
            wandb_config=None,
            use_wandb=False
        )

def distributed_run(
    main_func,
    worker_count,
    sweep_id=None,
    entity=None,
    project=None,
    args_dict=None,
    use_wandb=False
):
    """
    Spawns multiple processes, each executing the same main function
    in a distributed environment (or just single process if needed).
    """
    mp.set_start_method("spawn", force=True)
    processes = []

    for local_rank in range(worker_count):
        p = mp.Process(
            target=worker,
            args=(
                local_rank,
                worker_count,
                main_func,
                sweep_id,
                entity,
                project,
                args_dict,
                use_wandb
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def dprint(str_to_print): 
    '''
    Print only if the rank is 0 or if the code is running in a single node.
    '''
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        print(str_to_print)


def detect_environment():
    # Example 1: Check for an environment variable set on the cluster
    if 'SLURM_JOB_ID' in os.environ:  # Common in Slurm-managed clusters
        return "cluster"
    
    # Example 2: Use hostname to differentiate
    hostname = socket.gethostname()
    if "cluster_name" in hostname:  # Replace 'cluster_name' with part of your cluster's hostname
        return "cluster"
    
    # Default to local
    return "local"

def find_free_port():
    """
    References:
        - https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None, is_cuda_enabled=True):
    backend = 'nccl' if is_cuda_enabled else 'gloo'
    comp_env = detect_environment()
    if comp_env != 'local':
        # We are on a cluster
        os.environ['MASTER_PORT'] = '29501'
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = os.environ['SLURM_NODEID']
        node_list = os.environ['SLURM_NODELIST']
        master_node = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = master_node
        dist.init_process_group(backend=backend)
    else:  # We are on a PC
        os.environ['MASTER_ADDR'] = "localhost" if master_addr is None else master_addr
        os.environ['MASTER_PORT'] = master_port if master_port is not None else find_free_port()
        # Check if operating system is macOS
        if sys.platform == 'darwin':
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def send_shape(shape: list, dst: int, device=None):
    if device is None:
        device = torch.device('cuda') if dist.get_backend() == 'nccl' else torch.device('cpu')
    for s in shape:
        dist.send(tensor=torch.tensor(
            s, dtype=torch.int32).to(device), dst=dst)
    dist.send(tensor=torch.tensor(-1, dtype=torch.int32).to(device), dst=dst)


def receive_shape(src: int, device=None):
    # Rest of the code...
    if device is None:
        device = torch.device('cuda') if dist.get_backend(
        ) == 'nccl' else torch.device('cpu')
    shape = []
    temp = 0
    while True:
        temp = torch.tensor((0), dtype=torch.int32).to(device)
        dist.recv(tensor=temp, src=src)
        if temp == -1:
            break
        shape.append(temp.item())
    return shape

def check_gpus_per_rank():
    '''
    Ensure that the number of GPUs is the same on every rank in the distributed environment.
    This is necessary to perform tensor sharding. 
    '''
    # Get the number of GPUs available on the current rank
    local_gpus = torch.cuda.device_count()

    # Gather the number of GPUs from all ranks
    gpu_counts = [torch.tensor(0).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(gpu_counts, torch.tensor(local_gpus).cuda())

    # Convert gathered tensors to CPU and list
    gpu_counts = [gpu.item() for gpu in gpu_counts]

    # Check if all ranks have the same number of GPUs
    if len(set(gpu_counts)) != 1:
        raise ValueError("Mismatch in the number of GPUs across ranks")
    else:
        return gpu_counts[0]


def gather_node_info():
    # Get global rank and hostname
    global_rank = dist.get_rank()
    node_name = socket.gethostname()

    # Create a dictionary of {node_name: global_rank}
    local_info = {node_name: global_rank}

    # Gather the information from all ranks
    gathered_info = [None] * dist.get_world_size()  # Predefine the list size
    dist.all_gather_object(gathered_info, local_info)

    # Combine information into a dictionary where key is node_name and value is a list of global ranks
    node_rank_dict = {}
    for info in gathered_info:
        for node, rank in info.items():
            if node in node_rank_dict:
                node_rank_dict[node].append(rank)
            else:
                node_rank_dict[node] = [rank]

    # {'node1': [0, 1, 2], 'node2': [3, 4, 5], ...}  -> node 1 will have 3 gpus with global rank number 0, 1, 2

    # Sorting global ranks for each node
    for node in node_rank_dict:
        node_rank_dict[node].sort()

    return node_rank_dict

def broadcast_dict(d, src, group=None):
    l = [d]
    dist.broadcast_object_list(l, src=src, group=group)
    n = l[0]
    return n


def all_gather_dict(local_dict, group=None):
    """
    Gather dictionaries from all ranks and combine them into a list where each entry
    is the dictionary from a different rank.

    Args:
    - local_dict (dict): Dictionary on each rank to gather.

    Returns:
    - list of dict: List of dictionaries from all ranks.
    """
    # Serialize local_dict to a byte tensor
    serialized_dict = pickle.dumps(local_dict)
    tensor_dict = torch.ByteTensor(list(serialized_dict))

    # Determine the maximum length of serialized dictionaries across all ranks
    local_length = torch.tensor(
        [tensor_dict.size(0)], dtype=torch.int64, device=tensor_dict.device)
    max_length = torch.tensor([0], dtype=torch.int64,
                              device=tensor_dict.device)
    dist.all_reduce(local_length, op=dist.ReduceOp.MAX)
    max_length = local_length.item()

    # Pad tensor_dict to match the maximum length if necessary
    if tensor_dict.size(0) < max_length:
        tensor_dict = torch.cat([tensor_dict, torch.zeros(
            max_length - tensor_dict.size(0), dtype=torch.uint8)])

    # Gather the padded byte tensors from all ranks
    group_size = dist.get_world_size() if group is None else len(
        dist.get_process_group_ranks(group))
    gathered_tensors = [torch.empty(
        max_length, dtype=torch.uint8, device=tensor_dict.device) for _ in range(group_size)]
    dist.all_gather(gathered_tensors, tensor_dict, group=group)

    # Deserialize each gathered tensor into a dictionary
    gathered_dicts = [pickle.loads(bytes(t.tolist()))
                      for t in gathered_tensors]
    return gathered_dicts

