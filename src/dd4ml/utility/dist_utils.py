import os
import datetime
import sys
import pickle
import socket
import subprocess
from contextlib import closing
import torch
import torch.distributed as dist
import random


def _get_local_rank():
    """Get local rank, computed when needed to avoid issues with module loading."""
    return int(os.environ.get('SLURM_LOCALID', '0'))


def _get_device_for_backend():
    """Get appropriate device based on distributed backend."""
    if not dist.is_initialized():
        return torch.device('cpu')
    backend = dist.get_backend()
    if backend == 'nccl':
        return torch.device(f'cuda:{_get_local_rank()}')
    return torch.device('cpu')

def is_main_process():
    return int(os.environ.get("SLURM_PROCID", 0)) == 0

def dprint(to_print):
    '''
    Print only if the rank is 0 or if the code is running in a single node.
    '''
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        print(to_print)

def detect_environment():
    if 'SLURM_JOB_ID' in os.environ:
        return "cluster"
    hostname = socket.gethostname()
    if "cluster_name" in hostname:
        return "cluster"
    return "local"

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def get_shared_random_master_port(master_port=None, seed=42):
    if master_port is not None:
        return str(master_port)
    random.seed(seed)
    return str(random.randint(0, 6500))

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None, is_cuda_enabled=torch.cuda.is_available()):
    if dist.is_initialized():
        return
    
    backend = 'nccl' if is_cuda_enabled else 'gloo'
    comp_env = detect_environment()
    env_vars = {}
    multi_gpu = False
    if comp_env != 'local':  # SLURM cluster environment
        multi_gpu = is_cuda_enabled and torch.cuda.device_count() > 1
        env_vars['MASTER_PORT'] = get_shared_random_master_port(master_port, seed=12345) # TODO: Currently random so may not always be a free port. Whatever strategy you choose, make sure it is the same across all processes.
        env_vars['MASTER_ADDR'] = subprocess.getoutput(f"scontrol show hostname {os.environ.get('SLURM_NODELIST')} | head -n1")
        env_vars['WORLD_SIZE'] = os.environ.get('SLURM_NTASKS', '1')
        env_vars['RANK'] = os.environ.get('SLURM_PROCID', '0') if multi_gpu else os.environ.get('SLURM_NODEID', '0')
        env_vars['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')
        
        if is_cuda_enabled:
            torch.cuda.set_device(_get_local_rank())
        rank = int(env_vars['RANK'])
        world_size = int(env_vars['WORLD_SIZE'])
    else:  # Local environment
        env_vars['MASTER_ADDR'] = master_addr or "localhost"
        env_vars['MASTER_PORT'] = master_port or find_free_port()
        if sys.platform == 'darwin':
            env_vars["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Update environment variables
    os.environ.update(env_vars)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30))
    
    if multi_gpu: dprint("Multi-GPU environment detected.")

def send_shape(shape: list, dst: int, device=None):
    if device is None:
        device = _get_device_for_backend()

    # Pre-create tensors to avoid repeated allocation
    for s in shape:
        tensor = torch.tensor(s, dtype=torch.int32, device=device)
        dist.send(tensor=tensor, dst=dst)

    # Send termination signal
    end_tensor = torch.tensor(-1, dtype=torch.int32, device=device)
    dist.send(tensor=end_tensor, dst=dst)

def receive_shape(src: int, device=None):
    if device is None:
        device = _get_device_for_backend()

    shape = []
    # Pre-allocate tensor to avoid repeated allocation
    recv_tensor = torch.tensor(0, dtype=torch.int32, device=device)

    while True:
        dist.recv(tensor=recv_tensor, src=src)
        if recv_tensor.item() == -1:
            break
        shape.append(recv_tensor.item())

    return shape

def check_gpus_per_rank():
    world_size = dist.get_world_size()
    loc_gpus = torch.cuda.device_count()

    # Create tensors more efficiently
    local_gpu_tensor = torch.tensor(loc_gpus, device='cuda')
    gpu_counts = [torch.zeros_like(local_gpu_tensor) for _ in range(world_size)]

    dist.all_gather(gpu_counts, local_gpu_tensor)
    gpu_counts = [gpu.item() for gpu in gpu_counts]

    if len(set(gpu_counts)) != 1:
        raise ValueError("Mismatch in the number of GPUs across ranks")

    return gpu_counts[0]

def gather_node_info():
    glob_rank = dist.get_rank()
    node_name = socket.gethostname()
    loc_info = {node_name: glob_rank}
    gathered_info = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_info, loc_info)

    # Use defaultdict for more efficient dictionary operations
    from collections import defaultdict
    node_rank_dict = defaultdict(list)

    for info in gathered_info:
        for node, rank in info.items():
            node_rank_dict[node].append(rank)

    # Sort all rank lists and convert back to regular dict
    return {node: sorted(ranks) for node, ranks in node_rank_dict.items()}

def broadcast_dict(dictionary, src, group=None):
    obj_list = [dictionary]
    dist.broadcast_object_list(obj_list, src=src, group=group)
    broadcasted_dict = obj_list[0]
    return broadcasted_dict

def all_gather_dict(loc_dict, group=None):
    serialized_dict = pickle.dumps(loc_dict)
    tensor_dict = torch.frombuffer(serialized_dict, dtype=torch.uint8)

    # Get tensor length and find maximum across all processes
    loc_len = torch.tensor(tensor_dict.size(0), dtype=torch.int64, device=tensor_dict.device)
    max_len_tensor = loc_len.clone()
    dist.all_reduce(max_len_tensor, op=dist.ReduceOp.MAX)
    max_len = max_len_tensor.item()

    # Pad tensor to maximum length
    if tensor_dict.size(0) < max_len:
        padding = torch.zeros(max_len - tensor_dict.size(0), dtype=torch.uint8, device=tensor_dict.device)
        tensor_dict = torch.cat([tensor_dict, padding])

    # Gather tensors from all processes
    group_size = dist.get_world_size() if group is None else len(dist.get_process_group_ranks(group))
    gathered_tensors = [torch.empty(max_len, dtype=torch.uint8, device=tensor_dict.device)
                       for _ in range(group_size)]
    dist.all_gather(gathered_tensors, tensor_dict, group=group)

    # Deserialize gathered tensors back to dictionaries
    gathered_dicts = [pickle.loads(t.numpy().tobytes()) for t in gathered_tensors]
    return gathered_dicts
