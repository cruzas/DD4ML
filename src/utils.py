import os  
import torch
import subprocess
import torch.distributed as dist


def decide_tensor_device(ws, backend, gpu_id):
    if torch.cuda.is_available():
        if backend == 'gloo':
            if torch.cuda.device_count() < ws:
                return f'cuda:{gpu_id}'
            else:
                return f'cuda:{dist.get_rank()}'
        else:
            return f'cuda:{gpu_id}'
    else:
        return 'cpu'

def find_free_port():
    """
    References:
        - https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])  

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None, is_cuda_enabled=True):
    if not is_cuda_enabled:
        os.environ["CUDA_VISIBLE_DEVICES"]="" # TODO: Remove this line. It's just for debugging purposes.
    if rank is None and master_addr is None and master_port is None and world_size is None: # we are on a cluster
        print(f'Should be initializing {os.environ["SLURM_NNODES"]} nodes')
        ## Execute code on a cluster
        os.environ['MASTER_PORT'] = '29501'
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = os.environ['SLURM_NODEID']
        node_list = os.environ['SLURM_NODELIST']
        master_node = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1'
        )
        os.environ['MASTER_ADDR'] = master_node
        dist.init_process_group(backend='nccl')
    else: # To execute on a PC
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port if master_port is not None else find_free_port()
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        
def send_shape(shape: list, dst: int, device = None):
    if device is None:
        device = torch.device('cuda') if dist.get_backend() == 'nccl' else torch.device('cpu')
    for s in shape:
        dist.send(tensor=torch.tensor(s, dtype=torch.int32).to(device), dst=dst)
    dist.send(tensor=torch.tensor(-1, dtype=torch.int32).to(device), dst=dst)

def receive_shape(src: int, device = None):
    # Rest of the code...
    if device is None:
        device = torch.device('cuda') if dist.get_backend() == 'nccl' else torch.device('cpu')
    shape = []; temp = 0
    while True:
        temp = torch.tensor((0), dtype=torch.int32).to(device)
        dist.recv(tensor=temp, src=src)
        if temp == -1:
            break
        shape.append(temp.item())
    return shape

def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True, return_output=False, data_chunks_amount=1):
    """
    NOTE: Losses from different chunks are averaged.
    """
    # if model.__class__.__name__ != 'ParallelizedModel':
    #     raise ValueError('Model must be an instance of the "ParallelizedModel".')
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    if model.rank in model.all_stage_ranks[-1]:
        targets = targets.chunk(data_chunks_amount)
    # Compute loss
    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, data_chunks_amount=data_chunks_amount):
        if zero_grad:
            model.zero_grad()
        with torch.set_grad_enabled(compute_grad):
            outputs = model(inputs, chunks_amount=data_chunks_amount)
        # loss = torch.zeros(1).to(model.gpu_device)
        losses = [0] * data_chunks_amount
        loss = torch.tensor(0.0).to(model.tensor_device)
        if model.rank in model.all_stage_ranks[-1]:
            for i,out in enumerate(outputs):
                losses[i] = criterion(out, targets[i].to(out.device))
            loss = sum(losses)/len(losses)
        # Average losses across replicas
        if model.rank in model.all_stage_ranks[-1]:
            dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=model.layer_copies_group) # Summing the losses across final layers of each replicas
            loss = loss/model.tot_replicas
        loss_broadcast = dist.broadcast(tensor=loss.detach(), src=model.all_stage_ranks[-1][0], async_op=True) # Last layer of first model replica broadcasts the loss to all other ranks 
        if compute_grad and torch.is_grad_enabled():
            model.backward(losses)
        loss_broadcast.wait()
        if return_output:
            if model.rank in model.all_stage_ranks[-1]:
                # Returning outputs here in case we want to compute the accuracy afterwards
                return loss.item(), [output for output in outputs]
            else:
                return loss.item(), None
        return loss.item()
    return closure2 

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