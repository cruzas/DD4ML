import os
import datetime
loc_rank = os.environ.get('SLURM_LOCALID', '0')
import sys
import pickle
import socket
import subprocess
from contextlib import closing
import torch
import torch.distributed as dist
import random

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
        
        if is_cuda_enabled: torch.cuda.set_device(int(os.environ['SLURM_LOCALID']))
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
        device = torch.device(f'cuda:{loc_rank}') if dist.get_backend() == 'nccl' else torch.device('cpu')
    for s in shape:
        dist.send(tensor=torch.tensor(
            s, dtype=torch.int32).to(device), dst=dst)
    dist.send(tensor=torch.tensor(-1, dtype=torch.int32).to(device), dst=dst)

def receive_shape(src: int, device=None):
    if device is None:
        device = torch.device(f'cuda:{loc_rank}') if dist.get_backend() == 'nccl' else torch.device('cpu')
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
    loc_gpus = torch.cuda.device_count()
    gpu_counts = [torch.tensor(0).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(gpu_counts, torch.tensor(loc_gpus).cuda())
    gpu_counts = [gpu.item() for gpu in gpu_counts]
    if len(set(gpu_counts)) != 1:
        raise ValueError("Mismatch in the number of GPUs across ranks")
    else:
        return gpu_counts[0]

def gather_node_info():
    glob_rank = dist.get_rank()
    node_name = socket.gethostname()
    loc_info = {node_name: glob_rank}
    gathered_info = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_info, loc_info)
    node_rank_dict = {}
    for info in gathered_info:
        for node, rank in info.items():
            if node in node_rank_dict:
                node_rank_dict[node].append(rank)
            else:
                node_rank_dict[node] = [rank]
    for node in node_rank_dict:
        node_rank_dict[node].sort()
    return node_rank_dict

def broadcast_dict(d, src, group=None):
    l = [d]
    dist.broadcast_object_list(l, src=src, group=group)
    n = l[0]
    return n

def all_gather_dict(loc_dict, group=None):
    serialized_dict = pickle.dumps(loc_dict)
    tensor_dict = torch.ByteTensor(list(serialized_dict))
    loc_len = torch.tensor(
        [tensor_dict.size(0)], dtype=torch.int64, device=tensor_dict.device)
    max_len = torch.tensor([0], dtype=torch.int64,
                              device=tensor_dict.device)
    dist.all_reduce(loc_len, op=dist.ReduceOp.MAX)
    max_len = loc_len.item()
    if tensor_dict.size(0) < max_len:
        tensor_dict = torch.cat([tensor_dict, torch.zeros(
            max_len - tensor_dict.size(0), dtype=torch.uint8)])
    group_size = dist.get_world_size() if group is None else len(
        dist.get_process_group_ranks(group))
    gathered_tensors = [torch.empty(
        max_len, dtype=torch.uint8, device=tensor_dict.device) for _ in range(group_size)]
    dist.all_gather(gathered_tensors, tensor_dict, group=group)
    gathered_dicts = [pickle.loads(bytes(t.tolist()))
                      for t in gathered_tensors]
    return gathered_dicts
