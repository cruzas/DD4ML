import os  
import torch
import socket
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
    if model.model_handler.is_last_stage():
        targets = targets.chunk(data_chunks_amount)
    # Compute loss
    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, data_chunks_amount=data_chunks_amount, sync_loss='global'):
        '''
        sync_loss: 'global' or 'local' ('global' means every rank, 'local' means only the ranks within the same subdomain in data)
        '''
        if sync_loss not in ['global', 'local']:
            raise ValueError('sync_loss must be either "global" or "local".')
        if zero_grad:
            model.zero_grad()
        with torch.set_grad_enabled(compute_grad):
            outputs = model(inputs, chunks_amount=data_chunks_amount)
        losses = [0] * data_chunks_amount
        loss = torch.tensor(0.0).to(model.tensor_device)
        if model.model_handler.is_last_stage():
            for i, out in enumerate(outputs):
                losses[i] = criterion(out, targets[i].to(out.device))
            loss = torch.tensor((sum(losses)/len(losses)).item()).to(model.tensor_device)
        # Average losses across replicas
        if sync_loss == 'global':
            if model.model_handler.is_last_stage():
                dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=model.model_handler.get_layers_copy_group(mode='global')) # Summing the losses across final layers of each replicas
                loss = loss/model.model_handler.tot_replicas
            last_ranks = model.model_handler.get_stage_ranks(stage_name='last', mode='global')
            loss_broadcast = dist.broadcast(tensor=loss.detach(), src=last_ranks[0], group=model.model_handler.global_model_group, async_op=True) # each replica gets the average loss across all replicas (since we are averaging the losses first)
            # -> all subdomains will have the same loss
        else:
            if model.model_handler.is_last_stage():
                dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=model.model_handler.get_layers_copy_group(mode='local')) # Summing the losses across shard 0 of final layers of each replicas within the same subdomain
                loss = loss/model.num_replicas_per_subdomain
            last_stage_ranks = model.model_handler.get_stage_ranks(stage_name='last', mode='local')
            if len(last_stage_ranks) > 1:
                raise ValueError('Tensor sharding not implemented yet.')
            loss_broadcast = dist.broadcast(tensor=loss.detach(), src=last_stage_ranks[0], group=model.model_handler.get_sd_group(), async_op=True) # shard 0 of last layer of first model replica broadcasts the loss to all other replicas within the same subdomain
            # -> each subdomains may have a different loss
        
        if compute_grad and torch.is_grad_enabled():
            model.backward(losses)
        loss_broadcast.wait()
        if return_output:
            if model.model_handler.is_last_stage():
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

def list_flattener(l):
    '''
    Flattens a list of lists of lists of ... to a single list.
    '''
    while any(isinstance(i, list) for i in l):
        l = [item for sublist in l for item in sublist]
    return l
    