import json
import os
import pickle
import random
import socket
import subprocess
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


def dprint(str_to_print): 
    '''
    Print only if the rank is 0 or if the code is running in a single node.
    '''
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        print(str_to_print)


def get_starting_info(rank, base_file_name, epoch_file_name, num_epochs):
    starting_epoch = 0
    starting_num_iters = 0
    starting_network = ""
    epoch_results = []
    iter_results = []

    # Check if the model has already been trained
    max_epoch_already_trained = -1
    saved_networks = os.listdir('../saved_networks')
    for saved_network in saved_networks:
        if base_file_name in saved_network:
            saved_network_epoch = int(
                saved_network.split('_epoch_')[1].split('.pth')[0])

            if saved_network_epoch > max_epoch_already_trained:
                max_epoch_already_trained = saved_network_epoch
                starting_network = saved_network

    # Check that the corresponding csv files exist
    if max_epoch_already_trained > -1:
        if os.path.exists(epoch_file_name):
            starting_epoch = max_epoch_already_trained + 1
            if starting_epoch > num_epochs+1:
                if rank == 0:
                    print("Model already fully trained. Exiting...")
                exit(0)
            # Load epoch results
            df = pd.read_csv(epoch_file_name)
            epoch_results = df.to_dict('records')
            # Load iteration results
            df = pd.read_csv(epoch_file_name.replace('.csv', '_iter.csv'))
            iter_results = df.to_dict('records')
            # Get the number of iterations
            starting_num_iters = iter_results[-1]['iteration']
            # Print details
            if rank == 0 and max_epoch_already_trained > -1:
                print(
                    f"Model with parameters {base_file_name} already trained for {max_epoch_already_trained} epochs")
        else:
            starting_network = ""

    return starting_epoch, starting_num_iters, epoch_results, iter_results, starting_network


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
        # TODO: Remove this line. It's just for debugging purposes.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if rank is None and master_addr is None and master_port is None and world_size is None:  # we are on a cluster
        # print(f'Should be initializing {os.environ["SLURM_NNODES"]} nodes')
        # Execute code on a cluster
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
    else:  # To execute on a PC
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port if master_port is not None else find_free_port()
        dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)

def send_shape(shape: list, dst: int, device=None):
    if device is None:
        device = torch.device('cuda') if dist.get_backend(
        ) == 'nccl' else torch.device('cpu')
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


def closure(inputs, targets, criterion, model, compute_grad=True, zero_grad=True, return_output=False, data_chunks_amount=1, grad_norm_clip=None, outputs_only=False):
    """
    NOTE: Losses from different chunks are averaged.
    """
    # if model.__class__.__name__ != 'ParallelizedModel':
    #     raise ValueError('Model must be an instance of the "ParallelizedModel".')
    if isinstance(criterion, type):
        raise ValueError('Criterion must be an instance of a class.')
    if model.model_handler.is_last_stage() and targets is not None and not outputs_only:
        targets = targets.chunk(data_chunks_amount)
    # Compute loss

    def closure2(compute_grad=compute_grad, zero_grad=zero_grad, data_chunks_amount=data_chunks_amount, sync_loss='global', grad_norm_clip=grad_norm_clip, outputs_only=outputs_only):
        '''
        sync_loss: 'global' or 'local' ('global' means every rank, 'local' means only the ranks within the same subdomain in data)
        '''
        if sync_loss not in ['global', 'local']:
            raise ValueError('sync_loss must be either "global" or "local".')
        if zero_grad:
            model.zero_grad()
        with torch.set_grad_enabled(compute_grad):
            outputs = model(inputs, chunks_amount=data_chunks_amount)
            if outputs_only:
                return [output for output in outputs]
        losses = [0] * data_chunks_amount
        loss = torch.tensor(0.0).to(model.tensor_device)
        if model.model_handler.is_last_stage():
            for i, out in enumerate(outputs):
                losses[i] = criterion(out, targets[i].to(out.device))
            loss = torch.tensor(
                (sum(losses)/len(losses)).item()).to(model.tensor_device)
        # Average losses across replicas
        if sync_loss == 'global':
            if model.model_handler.is_last_stage():
                dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM, group=model.model_handler.get_layers_copy_group(
                    mode='global'))  # Summing the losses across final layers of each replicas
                loss = loss/model.model_handler.tot_replicas
            last_ranks = model.model_handler.get_stage_ranks(
                stage_name='last', mode='global')
            # each replica gets the average loss across all replicas (since we are averaging the losses first)
            loss_broadcast = dist.broadcast(tensor=loss.detach(
            ), src=last_ranks[0], group=model.model_handler.global_model_group, async_op=True)
            # -> all subdomains will have the same loss
        else:
            if model.model_handler.is_last_stage():
                # Summing the losses across shard 0 of final layers of each replicas within the same subdomain
                dist.all_reduce(tensor=loss, op=dist.ReduceOp.SUM,
                                group=model.model_handler.get_layers_copy_group(mode='local'))
                loss = loss/model.num_replicas_per_subdomain
            last_stage_ranks = model.model_handler.get_stage_ranks(
                stage_name='last', mode='local')
            if len(last_stage_ranks) > 1:
                raise ValueError('Tensor sharding not implemented yet.')
            # shard 0 of last layer of first model replica broadcasts the loss to all other replicas within the same subdomain
            loss_broadcast = dist.broadcast(tensor=loss.detach(
            ), src=last_stage_ranks[0], group=model.model_handler.get_sd_group(), async_op=True)
            # -> each subdomains may have a different loss

        if compute_grad and torch.is_grad_enabled():
            model.backward(losses)
            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip) 
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

# -----------------------------------------------------------------------------
# From minGPT utils
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        dict_config = config.to_dict()
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """Return a dict representation of the config with JSON-serializable values."""
        result = {}

        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                # Recursively call to_dict for nested CfgNode instances
                result[k] = v.to_dict()
            elif isinstance(v, type):  
                # Convert class references to string
                result[k] = f"{v.__module__}.{v.__name__}"
            elif v is None:
                # Replace None with a JSON-serializable equivalent
                result[k] = "null"
            elif v is Ellipsis:  
                # Replace ellipses (...) with a placeholder string
                result[k] = "..."
            else:
                # Keep other values as is
                result[k] = v

        return result


    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)

    def merge_and_cleanup(self):
        """
        Overwrite subdictionary values with corresponding upper-level values 
        and remove redundant upper-level keys.
        """
        
        # Get dictionary keys
        keys = list(self.__dict__.keys())

        # Keys to look into
        keys_to_look = ["system", "data", "model", "trainer"]

        # Keys other than keys_to_look
        other_keys = [k for k in keys if k not in keys_to_look]

        # Merge and cleanup
        for ok in other_keys:
            delete_ok = False
            for k in keys_to_look:
                # Check if ok is a key in self.__dict__[k]
                if ok in self.__dict__[k].__dict__:
                    # Overwrite the value
                    self.__dict__[k].__dict__[ok] = self.__dict__[ok]
                    delete_ok = True
            # Remove the key
            if delete_ok: del self.__dict__[ok]
           
            