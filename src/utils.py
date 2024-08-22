import os  
import torch
import subprocess
import torch.distributed as dist

def find_free_port():
    """
    Returns a free port number on localhost.

    This function binds a socket to an available port on localhost and returns the port number.

    Returns:
        str: The free port number.

    References:
        - https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])  

def prepare_distributed_environment(rank=None, master_addr=None, master_port=None, world_size=None):
    """
    Initializes the distributed environment for running code on a cluster or a PC.

    Args:
        rank (int, optional): The rank of the current process. Defaults to None.
        master_addr (str, optional): The address of the master node. Defaults to None.
        master_port (str, optional): The port number on the master node. Defaults to None.
        world_size (int, optional): The total number of processes in the distributed environment. Defaults to None.
    """
    if rank is None and master_addr is None and master_port is None and world_size is None: # we are on a cluster
        print(f'Should be initializing {os.environ["SLURM_NNODES"]} nodes')
        ## Execute code on a cluster
        os.environ["MASTER_PORT"] = "29501"
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NNODES"]
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = os.environ["SLURM_NODEID"]
        node_list = os.environ["SLURM_NODELIST"]
        master_node = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1"
        )
        os.environ["MASTER_ADDR"] = master_node
        dist.init_process_group(backend="nccl")
    else: # To execute on a PC
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port 
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def send_shape(shape: list, dst: int, device = None):
    """
    Sends a list of shapes to a destination process.

    Args:
        shape (list): A list of shapes to be sent.
        dst (int): The destination process to send the shapes to.
        device (torch.device, optional): The device to send the shapes to. If not provided, it will be determined based on the backend. Defaults to None.
    """
    if device is None:
        device = torch.device('cuda') if dist.get_backend() == 'nccl' else torch.device('cpu')
    for s in shape:
        dist.send(tensor=torch.tensor(s, dtype=torch.int32).to(device), dst=dst)
    dist.send(tensor=torch.tensor(-1, dtype=torch.int32).to(device), dst=dst)
        
def receive_shape(src: int, device = None):
    """
    Receive the shape of a tensor from a specified source.

    Args:
        src (int): The source from which to receive the tensor shape.
        device (torch.device, optional): The device on which to place the tensor. 
            If not provided, the default device will be used.

    Returns:
        list: The shape of the received tensor.

    """
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
    Compute the closure function for training a parallelized model.

    Args:
        inputs (Tensor): The input data.
        targets (Tensor): The target data.
        criterion (class): The loss criterion class.
        model (Parallelized_Model): The parallelized model instance.
        compute_grad (bool, optional): Whether to compute gradients. Defaults to True.
        zero_grad (bool, optional): Whether to zero gradients. Defaults to True.
        return_output (bool, optional): Whether to return the model outputs. Defaults to False.
        data_chunks_amount (int, optional): The number of data chunks. Defaults to 1.

    Returns:
        closure2 (function): The closure function for training the model.

    NOTE: Losses from different chunks are averaged.
    """
    if model.__class__.__name__ != 'Parallelized_Model':
        raise ValueError('Model must be an instance of the "Parallelized_Model".')
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    if model.rank == model.rank_list[-1]:
        targets = targets.chunk(data_chunks_amount)
    # Compute loss
    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, data_chunks_amount=data_chunks_amount):
        if zero_grad:
            model.zero_grad()
        with torch.set_grad_enabled(compute_grad):
            outputs = model(inputs, chunks_amount=data_chunks_amount)
        # loss = torch.zeros(1).to(model.gpu_device)
        losses = [0]*data_chunks_amount
        loss = torch.tensor(0.0).to(model.tensor_device)
        if model.rank == model.rank_list[-1]:
            for i,out in enumerate(outputs):
                losses[i] = criterion(out, targets[i].to(out.device))
            loss = sum(losses)/len(losses)
        # Average losses across replicas
        if model.rank == model.rank_list[-1]:
            dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=model.final_layer_group) # Summing the losses across final layers of each replicas
            loss = loss/model.num_replicas
        loss_broadcast = dist.broadcast(tensor=loss.detach(), src=model.model_ranks[0][-1], async_op=True) # Last layer of first model replica broadcasts the loss to all other ranks 
        if compute_grad and torch.is_grad_enabled():
            model.backward(losses)
            # Synchronize model gradients
            model.sync_grads()
        loss_broadcast.wait()
        if return_output:
            if model.rank == model.rank_list[-1]:
                # Returning outputs here in case we want to compute the accuracy afterwards
                return loss.item(), [output for output in outputs]
            else:
                return loss.item(), None
        return loss.item()
    return closure2 
