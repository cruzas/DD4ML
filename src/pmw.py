# Parallelized Model Wrapper (PMW) for PyTorch

import math
import time 
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
from utils import *
    
# TODO: Currently, backend_device for many things is 'cuda:0'. This assumed the Piz Daint infrastructure, in which each had only one GPU.
# In the future, we need to make this more generic and consider the case where each node has multiple GPUs.
class Parallelized_Model(nn.Module):
    '''
    Data parallel and weight parallel model wrapper.
    Meaning of the parameters:
    - stage_list: List of tuples. Each tuple contains a layer class and its parameters. The layer class is a function that returns a layer (e.g. nn.Linear, nn.Conv2d, nn.Sequential, etc.). The parameters are the parameters of the layer class.
    '''
    def __init__(self, stage_list, sample, num_replicas=1, device=None, data_parallel=False):
        super(Parallelized_Model, self).__init__()
        if num_replicas <= 1 and data_parallel:
            raise ValueError("Data parallelism requires at least two replicas.")

        self.num_replicas = num_replicas
        self.world_size = dist.get_world_size()
        self.data_parallel = data_parallel
        # self.backend_device = 'cpu' if dist.get_backend() == 'gloo' else 'cuda:0' # TODO: remove if not used
        self.tensor_device = decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0) if device is None else device
        if num_replicas*len(stage_list) != self.world_size:
            raise ValueError(f"The number of replicas times the number of layers ({num_replicas}*{len(stage_list)}={num_replicas*len(stage_list)}) must be equal to the world size ({self.world_size}).")
        self.rank = dist.get_rank()
        self.final_layer_group = dist.new_group(ranks=[r for r in range(self.world_size) if r%len(stage_list) == len(stage_list)-1], use_local_synchronization=True)

        # Model copy rank list
        self.stage_list = stage_list
        self.model_ranks = [[r+k*len(self.stage_list) for r in range(len(self.stage_list))] for k in range(num_replicas)] 
        for model_rank in self.model_ranks:
            if self.rank in model_rank:
                self.rank_list = model_rank
                self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
                break
        for ranks in self.model_ranks:
            if self.rank in ranks:
                # TODO: Change gpu_id to be more generic for tensor sharding later on...
                self.model = Weight_Parallelized_Model(stage_list=stage_list, rank_list=ranks, sample=sample, device=device)
                self.subdomain = self.model.subdomain
        
        # Create a process group for each layer. This group contains all the ranks that are responsible for the layer across all replicas.
        for layer_idx in range(len(self.stage_list)):
            # Collect ranks that are responsible for this layer across all replicas
            ranks = [r + layer_idx for r in range(0, self.world_size, len(self.stage_list))]
            # Create a new group containing these ranks
            if self.rank in ranks:
                self.layer_copies_group = dist.new_group(ranks, use_local_synchronization=True)
        self.sync_params()
    
    def subdomain_forward(self, sync=False):
        """
        Performs forward pass on the subdomain.

        Parameters:
            sync (bool): If True, performs synchronous forward pass by reducing outputs using `dist.all_reduce`.
                         Defaults to False.

        Returns:
            outputs (list): List of outputs from the forward pass.
        """
        outputs = self.subdomain.forward()
        if sync: # NOTE: Probably never used, but included for completeness
            with torch.no_grad():
                for output in outputs:
                    dist.all_reduce(output, group=self.final_layer_group, op=dist.ReduceOp.SUM)
        return outputs
    
    def subdomain_backward(self, sync=True):
        """
        Backward pass through the subdomain.

        Parameters:
            sync (bool, optional): Whether to synchronize gradients when using data parallel settings. Defaults to True.

        Returns:
            None
        """
        self.subdomain.backward()
        if sync and self.data_parallel: # NOTE: Sync True for sure for data parallel settings, but leaving the option to the user to use False
            self.sync_grads() 

    def subdomain_params(self):
        """
        Returns the parameters of the subdomain.

        Returns:
            dict: A dictionary containing the parameters of the subdomain.
        """
        return self.subdomain.parameters()
        
    def subdomain_grad(self):
        """
        Returns the gradient of the subdomain.

        Returns:
            The gradient of the subdomain.
        """
        return self.subdomain.grad()
    
    def subdomain_grad_norm(self, p=2):
        """
        Calculate the gradient norm of the subdomain.

        Parameters:
        - p (int, optional): The norm type. Defaults to 2.

        Returns:
        - float: The gradient norm of the subdomain.
        """
        return self.subdomain.grad_norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        """
        Returns the global parameters of the model.

        Parameters:
            clone (bool): Whether to clone the parameters or not. Default is False.

        Returns:
            torch.nn.parameter.Parameter: The global parameters of the model.
        """
        return self.model.parameters(clone=clone)
    
    def parameters_norm(self, p=2): # Returns the global parameters norm of the model
        """
        Returns the global parameters norm of the model.

        Parameters:
        - p (int): The norm type. Default is 2.

        Returns:
        - float: The global parameters norm of the model.
        """
        return self.model.parameters().norm(p=p)
    
    def grad(self, clone=False): # Returns the global gradient of the model
        """
        Returns the global gradient of the model.

        Parameters:
            clone (bool): Whether to clone the model before computing the gradient. Default is False.

        Returns:
            ndarray: The global gradient of the model.
        """
        return self.model.grad(clone=clone)

    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        """
        Returns the global gradient norm of the model.

        Parameters:
        - p (int): The norm type. Default is 2.

        Returns:
        - float: The global gradient norm of the model.
        """
        return self.model.grad_norm(p=p)
        
    def subdomain_grad_norm(self, p=2): # Returns the subdomain gradient norm of the model
        """
        Returns the subdomain gradient norm of the model.

        Parameters:
        - p (int, optional): The norm type. Defaults to 2.

        Returns:
        - float: The subdomain gradient norm of the model.
        """
        return self.model.subdomain_grad_norm(p=p)

    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        """
        Forward pass through the model.

        Args:
            x: The input data.
            chunks_amount: The number of chunks to split the input data into.
            reset_grad: Whether to reset the gradients of the model parameters.
            compute_grad: Whether to compute gradients during the forward pass.

        Returns:
            The output of the forward pass.
        """
        return self.model.forward(x, chunks_amount=chunks_amount, reset_grad=reset_grad, compute_grad=compute_grad)
    
    def backward(self, losses, sync=True):
        """
        Backpropagates the given losses through the model and updates the gradients.

        Parameters:
        - losses (Tensor): The loss values to backpropagate.
        - sync (bool): Whether to synchronize the gradients across all replicas. Default is True.

        Returns:
        None
        """
        self.model.backward(losses=losses)
        if sync: # Synchronize the gradients across all replicas (True by default since this will always be done in both data parallel approaches)
            self.sync_grads() 
    
    def sync_params(self, method='average'):
        """
        Synchronizes the parameters of the model across all replicas.

        Parameters:
            method (str): The method to use for synchronization. Default is 'average'.
                - 'average': Divides the summed parameters by the number of replicas.
                - 'sum': No operation is performed since the parameters are already summed.

        Raises:
            ValueError: If the method is not supported.

        Returns:
            None
        """
        for param in self.model.parameters():
            dist.all_reduce(tensor=param.data, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
            if method == 'average':
                param.data /= self.num_replicas
            elif method == 'sum':
                pass # nothing to do since we already summed the parameters through all_reduce
            else:
                raise ValueError(f"Method {method} is not supported.")

    def sync_grads(self):
        """
        Synchronizes gradients across multiple replicas of the model.

        This method performs gradient synchronization by reducing the gradients of each parameter
        across all replicas using the `all_reduce` function from the `torch.distributed` module.
        The reduction operation used is the sum (`ReduceOp.SUM`).

        After the reduction, the gradients are divided by the number of replicas (`self.num_replicas`)
        to obtain the average gradient.

        Note:
        - This method assumes that the model's parameters are already initialized.
        - The synchronization is performed using the `layer_copies_group` group.

        Args:
            self (object): The `self` object.
        
        Returns:
            None
        """
        for param in self.model.parameters():
            dist.all_reduce(tensor=param.grad, group=self.layer_copies_group, op=dist.ReduceOp.SUM)
            param.grad /= self.num_replicas

# Global model class
class Weight_Parallelized_Model(nn.Module):
    def __init__(self, stage_list, rank_list, sample, device=None):
        '''Initializes the Weight_Parallelized_Model class. This class is used to parallelize the weights of a model across multiple ranks.
        Parallelization in weights includes pipelining the forward and backward passes across multiple ranks and layer sharding.

        Parameters:
        - stage_list (list): A list of tuples containing the layer class and its parameters.
        - rank_list (list): A list of ranks for distributed training.
        - sample (torch.Tensor): The input sample.
        - device (torch.device): The device to be used for computation. Default is None.
        Attributes:
        - rank (int): The rank of the current process.
        - rank_list (list): The list of ranks for distributed training.
        - rank_index (int): The index of the current rank in the rank_list.
        - master_group (torch.distributed.ProcessGroup): The process group for synchronization.
        - backend (str): The backend for distributed training.
        - backend_device (str): The device to be used for computation.
        - tensor_device (torch.device): The device to be used for tensor operations.
        - inputs (list): A list to store the inputs of the next rank or layer.
        - outputs (list): A list to store the outputs of the previous rank or layer.
        - grad_output (list): A list to store the gradients of the output of the current layer.
        - stage (torch.nn.Module): The current layer of the model.
        - num_f_evals (int): The number of forward evaluations.
        - num_g_evals (int): The number of gradient evaluations.
        - f_time (float): The time taken for the forward pass.
        - g_time (float): The time taken for gradient computation.
        - subdomain (Weight_Parallelized_Subdomain): The subdomain model.
        '''
        
        '''
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the stage_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        ''' 
        super(Weight_Parallelized_Model, self).__init__()
        self.rank = dist.get_rank()
        self.rank_list = rank_list
        self.rank_index = rank_list.index(self.rank)
        self.master_group = dist.new_group(ranks=self.rank_list, use_local_synchronization=True)
        self.backend = dist.get_backend()
        self.backend_device = 'cpu' if self.backend == 'gloo' else self.tensor_device
        self.tensor_device = decide_tensor_device(ws=dist.get_world_size(), backend=dist.get_backend(), gpu_id=0) if device is None else device
        self.inputs = []  # Each rank will store here the input of the next rank or layer (so the output of the current layer)  | -> this is needed for the backward pass
        self.outputs = []  # Each rank will store here the output of the previous rank or layer (so its input)                  | -> this is needed for the backward pass
        self.grad_output = [] # Each rank will store here the gradient of the output of the current layer (so the gradient of the loss w.r.t. the output of the current layer) | -> this is needed for the backward pass
        (layer_class, params) = stage_list[self.rank_index]
        self.stage = layer_class(**params).to(self.tensor_device) # Initialize the layer with provided parameters
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations
        self.f_time = 0 # Forward pass time
        self.g_time = 0 # Gradient computation time
        # This is built during forward/backward passes in the initialization phase
        # self.shapes 
    
        # Setup phase to compute the shapes of the tensors in the pipeline
        self.setup_phase = True
        loss = None
        out = self.forward(torch.randn(*sample.shape).to(self.tensor_device), chunks_amount=1, reset_grad=True, compute_grad=True)
        if self.rank == rank_list[-1]:
            loss = nn.MSELoss()(out[0], torch.rand_like(out[0]))
        self.backward([loss])
        self.zero_grad()
        self.setup_phase = False
        # End of setup phase
        self.subdomain = Weight_Parallelized_Subdomain(self) # Initialize the subdomain model

    def zero_counters(self):
        """
        Resets all the counters and timers to zero.
        """
        self.num_f_evals = 0
        self.num_g_evals = 0
        self.f_time = 0
        self.g_time = 0
        self.subdomain.num_f_evals = 0
        self.subdomain.num_g_evals = 0
        self.subdomain.f_time = 0
        self.subdomain.g_time = 0

    def zero_grad(self):
        """
        Clears the gradients of all parameters in the model.

        This method sets the gradients of all parameters in the model to zero.
        It is typically called before the backward pass to avoid accumulating gradients from previous iterations.

        Parameters:
            self (object): The current instance of the class.

        Returns:
            None
        """
        self.stage.zero_grad()
    
    def grad(self, clone=False): # Returns the global gradient of the model
        """
        Returns the global gradient of the model.

        Parameters:
            clone (bool): Whether to clone the gradient tensors or not. Default is False.

        Returns:
            Weight_Parallelized_Tensor: The global gradient of the model.
        """
        gradient = [param.grad.clone() if clone else param.grad for param in self.parameters()]
        return Weight_Parallelized_Tensor(gradient, self.backend, self.master_group, self.rank)
    
    def grad_norm(self, p=2): # Returns the global gradient norm of the model
        """
        Returns the global gradient norm of the model.

        Parameters:
        - p (int, optional): The norm type. Default is 2.

        Returns:
        - float: The global gradient norm of the model.
        """
        return self.grad().norm(p=p)
    
    def parameters(self, clone=False): # Returns the global parameters of the model
        """
        Returns the global parameters of the model.

        Parameters:
            clone (bool): Whether to clone the parameters or not. Default is False.

        Returns:
            Weight_Parallelized_Tensor: A tensor containing the global parameters.
        """
        params = [param.clone() if clone else param for param in self.stage.parameters()]
        return Weight_Parallelized_Tensor(params, self.backend, self.master_group, self.rank)
    
    def subdomain_grad_norm(self, p=2): # Returns the subdomain gradient norm of the model
        """
        Returns the subdomain gradient norm of the model.

        Parameters:
        - p (int, optional): The norm type. Default is 2.

        Returns:
        - float: The subdomain gradient norm of the model.
        """
        return torch.norm(torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p).item()

    def forward(self, x, chunks_amount=1, reset_grad = False, compute_grad = True):
        """
        Forward pass through the pipeline.
        Args:
            x (torch.Tensor): Input tensor.
            chunks_amount (int, optional): Number of chunks to split the input tensor into. Defaults to 1.
            reset_grad (bool, optional): Flag to reset the gradients of the model before starting to accumulate them again. Defaults to False.
            compute_grad (bool, optional): Flag to compute gradients. Defaults to True.
        Returns:
            torch.Tensor or bool: Output tensor if the rank is in the last layer's rank list, else True.
        """
        
        start = time.time()
        # Initialize the input and output tensors (needed for the backward pass)
        self.inputs = [None]*chunks_amount 
        self.outputs = [None]*chunks_amount  
        compute_grad = False if not torch.is_grad_enabled() else compute_grad # flag to avoid storing the tensors needed to compute the gradients
        self.num_f_evals += 1 
        with torch.set_grad_enabled(compute_grad):
            if reset_grad:
                self.zero_grad() # Reset the gradients of the model before starting to accumulate them again
                
            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(chunks_amount, dtype=torch.int32).to(self.tensor_device)
            if self.rank == self.rank_list[0]: # If the rank is in the first layer's rank list, send the input to the next device
                chunks = x.chunk(chunks_amount) # Chunkenize the input tensor
                self.inputs = chunks if compute_grad else [] 
                chunk_shapes = torch.tensor([chunk.shape[0] for chunk in chunks], dtype=torch.int32).to(self.backend_device) # Store the batch size of each chunk
            # Broadcast the chunk_shapes tensor to all the ranks (this allows to know the shape of the input tensor for each rank in the pipeline and prepare the recv function)
            shape_transfer = dist.broadcast(tensor=chunk_shapes, src=self.rank_list[0], group=self.master_group, async_op=True) # broadcasting only the batch size | async operation to avoid blocking the first layer

            # Go through the pipeline
            for c in range(chunks_amount):
                if self.rank_index == 0: # Beginning of the pipeline (first layer)
                    chunk = chunks[c].to(self.tensor_device)
                    out = self.stage.forward(chunk) 
                    next_rank = self.rank_list[self.rank_index+1]
                    self.outputs[c] = out if compute_grad else None
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(chunks[c].shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device=self.backend_device)
                    dist.send(tensor=out.to(self.backend_device), dst=next_rank) # send the tensor
                elif self.rank_index == len(self.rank_list)-1: # End of the pipeline (last layer)
                    shape_transfer.wait() # wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = receive_shape(src=self.rank_list[self.rank_index-1], device=self.backend_device)
                        temp = torch.empty(*shapes, device=self.backend_device, requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device=self.backend_device, requires_grad=True)
                    dist.recv(tensor=temp, src=self.rank_list[self.rank_index-1])
                    temp = temp.to(self.tensor_device)
                    self.inputs[c] = temp if compute_grad else None
                    out = self.stage.forward(temp)
                    self.outputs[c] = out 
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                else: # Middle of the pipeline (between the first and the last layer)
                    shape_transfer.wait() # Wait for the shape to be broadcasted
                    if self.setup_phase:
                        shapes = receive_shape(src=self.rank_list[self.rank_index-1], device=self.backend_device)
                        temp = torch.empty(*shapes, device=self.backend_device, requires_grad=True)
                    else:
                        temp = torch.empty(*self.shapes[0](chunk_shapes[c]), device=self.backend_device, requires_grad=True)
                    dist.recv(tensor=temp, src=self.rank_list[self.rank_index-1])
                    temp = temp.to(self.tensor_device)
                    out = self.stage.forward(temp)
                    self.inputs[c] = temp if compute_grad else None
                    self.outputs[c] = out if compute_grad else None
                    next_rank = self.rank_list[self.rank_index+1]
                    if self.setup_phase:
                        input_shape = lambda x: [x]+list(temp.shape)[1:]
                        output_shape = lambda x: [x]+list(out.shape)[1:]
                        self.shapes = [input_shape, output_shape]
                        send_shape(out.shape, dst=next_rank, device=self.backend_device)
                    dist.send(tensor=out.to(self.backend_device), dst=next_rank) # send the tensor
        self.f_time += time.time() - start
        # If the rank is in the last layer's rank list, return the output, else True (for some reason we can't remember)
        return self.outputs if self.rank == self.rank_list[-1] else True

    def backward(self, losses):
        """
        Performs the backward pass of the pipeline model.
        Args:
            losses (list): A list of loss values for each chunk.
        Returns:
            None
        """
        
        chunks_amount = len(losses) # Number of chunks
        self.grad_output = [None]*chunks_amount 
        start = time.time()
        self.num_g_evals += 1 
        for c, loss in enumerate(losses): # Chunked loss
            chunk_size = self.outputs[c].shape[0]
            if self.rank == self.rank_list[-1]: # End of the pipeline
                self.grad_output[c] = autograd.grad(loss, self.outputs[c], retain_graph=True)[0]     # TODO: Update so that it takes into account sequential models
                grad_data = autograd.grad(self.outputs[c], self.inputs[c], grad_outputs=self.grad_output[c], retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                dist.send(tensor=grad_data.to(self.backend_device), dst=self.rank_list[-2]) # TODO make this async if possible
                for param in self.stage.parameters():
                    if param.grad is None:
                        param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
            elif self.rank == self.rank_list[0]: # Beginning of the pipeline
                self.grad_output[c] = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device, requires_grad=True)
                dist.recv(self.grad_output[c], src=self.rank_list[1])       
                self.grad_output[c] = self.grad_output[c].to(self.tensor_device).detach()
                for param in self.stage.parameters():
                    if param.grad is None:
                        param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
            else: # Middle of the pipeline
                self.grad_output[c] = torch.empty(*self.shapes[1](chunk_size), device=self.backend_device, requires_grad=True)
                dist.recv(self.grad_output[c], src=self.rank_list[self.rank_index+1])       
                self.grad_output[c] = self.grad_output[c].to(self.tensor_device).detach()
                data_grad2 = autograd.grad(self.outputs[c], self.inputs[c], grad_outputs=self.grad_output[c], retain_graph=True)[0] # this is needed to compute the derivative at the previous stage
                dist.send(tensor=data_grad2.to(self.backend_device), dst=self.rank_list[self.rank_index-1]) # TODO make this async if possible
                for param in self.stage.parameters():
                    if param.grad is None: 
                        param.grad = autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
                    else:
                        param.grad += autograd.grad(self.outputs[c], param, grad_outputs=self.grad_output[c], retain_graph=True)[0]/chunks_amount
        # TODO / NOTE: maybe we can delete self.inputs to free memory. It is not used anymore after the backward pass. (even in subdomains)
        self.g_time += time.time() - start
        return None
    
# TODO: We may need a "Parallelized_Subdomain" class which includes data parallelism and weight parallelism. 
    
# Subdomain model class
class Weight_Parallelized_Subdomain(nn.Module):
    def __init__(self, model):
        """
        A module for weight parallelization of subdomains.
        Args:
            model (Weight_Parallelized_Model): The entire Weight_Parallelized_Model structure.
        Attributes:
            model (Weight_Parallelized_Model): The entire Weight_Parallelized_Model structure.
            num_f_evals (int): Number of forward evaluations.
            num_g_evals (int): Number of gradient evaluations.
            f_time (float): Time taken for forward evaluations.
            g_time (float): Time taken for gradient evaluations.
        Methods:
            forward(): Performs forward pass through the model.
            backward(): Performs backward pass through the model.
            grad(): Returns the gradients of the model parameters.
            grad_norm(): Returns the norm of the gradients of the model parameters.
        """
        super(Weight_Parallelized_Subdomain, self).__init__()
        self.model = model
        self.num_f_evals = 0 # Number of forward evaluations
        self.num_g_evals = 0 # Number of gradient evaluations   
        self.f_time = 0
        self.g_time = 0
        
    def forward(self):
        """
        Performs forward pass through the model.
        Returns:
            torch.Tensor: The model outputs.
        """
        start = time.time()
        for k, chunk in enumerate(self.model.inputs):
            self.model.outputs[k] = self.model.stage(chunk)
        self.num_f_evals += 1
        self.f_time += time.time() - start
        return self.model.outputs

    def backward(self):
        """
        Performs backward pass through the model.
        """
        start = time.time()
        for k in range(len(self.model.outputs)):
            for param in self.model.stage.parameters():
                if param.grad is None:
                    param.grad = autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
                else:
                    param.grad += autograd.grad(self.model.outputs[k], param, grad_outputs=self.model.grad_output[k], retain_graph=True)[0]/len(self.model.outputs)
        self.num_g_evals += 1
        self.g_time += time.time() - start
            
    def grad(self):
        """
        Returns the gradients of the model parameters.
        Returns:
            List[torch.Tensor]: The gradients of the model parameters.
        """
        return [param.grad for param in self.model.parameters()]
    
    def grad_norm(self):
        """
        Returns the norm of the gradients of the model parameters.
        Returns:
            float: The norm of the gradients of the model parameters.
        """
        return torch.norm(torch.cat([param.grad.flatten() for param in self.model.parameters()], dim=0), p=2).item()

# TODO: We may need a "Parallelized_Tensor" class which includes data parallelism of tensors (in case we want to do different strategies with the steps in APTS).

# Global gradient class
class Weight_Parallelized_Tensor(nn.Module):
    """
    A custom PyTorch module for parallelized tensor operations with gradient accumulation.
    Args:
        tensor (list): List of tensors representing the model parameters.
        backend (str): The backend used for parallelization (e.g., 'gloo', 'nccl').
        master_group (dist.ProcessGroup): The process group containing all ranks related to the model.
        rank (int): The rank of the current process.
    Methods:
        norm(p=2):
            Computes the L2 norm of the tensor.
        __iter__():
            Returns an iterator over the tensor.
        __repr__():
            Returns a string representation of the object.
        __matmul__(a):
            Performs matrix multiplication between the tensor and another tensor a.
        __rmatmul__(a):
            Performs matrix multiplication between another tensor a and the tensor.
        __rmul__(a):
            Performs element-wise multiplication between a scalar or tensor a and the tensor.
        __mul__(a):
            Performs element-wise multiplication between the tensor and another tensor a.
        __add__(a):
            Performs element-wise addition between the tensor and another tensor a.
        __sub__(a):
            Performs element-wise subtraction between the tensor and another tensor a.
    """
    def __init__(self, tensor, backend, master_group, rank):
        super().__init__()  # Call to the superclass (nn.Module) constructor
        self.tensor = tensor
        self.backend = backend
        self.master_group = master_group
        self.rank = rank
        
    def norm(self, p=2):
        if p == 2:
            return math.sqrt(self @ self)
        else:
            # Implement generic p
            raise NotImplementedError("Only L2 norm is implemented.")
    
    def __iter__(self):
        return iter(self.tensor)
    
    def __repr__(self):
        return f'Rank {self.rank}\nGradient: {self.model.subdomain.grad()}'
    
    def __matmul__(self, a): # self.grad @ a
        return self * a
    
    def __rmatmul__(self, a): # a @ self.grad
        return a * self

    def __rmul__(self, a):  # a * self.grad
        return self.__mul__(a)  # This handles the commutative property of multiplication
    
    def __mul__(self, a):  # self.grad * a
        """
        Multiply the Weight_Parallelized_Tensor instance by a scalar or another Weight_Parallelized_Tensor instance.

        Parameters:
            a (Union[Weight_Parallelized_Tensor, float, int]): The scalar or Weight_Parallelized_Tensor instance to multiply with.

        Returns:
            Union[float, Weight_Parallelized_Tensor]: If `a` is a Weight_Parallelized_Tensor instance, the method returns the dot product of the flattened gradients of `self` and `a` after reducing the gradients on the master rank. If `a` is a scalar, the method returns a new Weight_Parallelized_Tensor instance with each element multiplied by `a`.
        """
        if isinstance(a, Weight_Parallelized_Tensor):  # When both operands are Weight_Parallelized_Tensor instances
            g1 = torch.cat([p.flatten() for p in self.tensor], dim=0)  # Flatten the gradients
            g2 = torch.cat([p.flatten() for p in a.tensor], dim=0)  # Flatten the gradients
            g3 = g1 @ g2
            g3 = g3.to(f'cpu' if self.backend == 'gloo' else f'cuda:0')
            dist.all_reduce(tensor=g3, group=self.master_group, op=dist.ReduceOp.SUM)  # Sum the gradients on the master rank
            return g3.item()
        else:
            return Weight_Parallelized_Tensor([p*a for p in self.tensor], backend=self.backend, master_group=self.master_group, rank=self.rank)   # Multiply model by a scalar or tensor

    def __add__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p+q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)
    
    def __sub__(self, a):
        if isinstance(a, Weight_Parallelized_Tensor):
            return Weight_Parallelized_Tensor([p-q for p,q in zip(self.tensor, a.tensor)], backend=self.backend, master_group=self.master_group, rank=self.rank)

def decide_tensor_device(ws, backend, gpu_id):
    """
    Determines the device on which to perform tensor operations.

    Parameters:
    - ws (int): The number of worker processes.
    - backend (str): The backend for distributed training.
    - gpu_id (int): The ID of the GPU to use.

    Returns:
    - str: The device on which to perform tensor operations. It can be 'cpu' or 'cuda:<gpu_id>'.

    """
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


# NOTE: TO BE DONE
class TensorSharding(nn.Module):
    '''
    manually define sharding strategy for a layer (depending on the NN type).  
    '''
    def __init__(self, module, rank_list, device_mesh, parallelize_plan):
        super(TensorSharding, self).__init__()
        self.module = module
        self.rank_list = rank_list
        self.device_mesh = device_mesh
        self.parallelize_plan = parallelize_plan
        

        for i, (layer, ranks) in enumerate(zip(module, rank_list)):
            if len(ranks) == 1:
                # module[i] = parallelize_module(layer, device_mesh, parallelize_plan)
                raise NotImplementedError("Tensor parallelism is not yet implemented.")
            else:
                raise NotImplementedError("Tensor parallelism is not yet implemented.")
        return module

    def unshard(self, rank=None): # or to_local
        '''
        Send all shards to the rank specified.
        '''
        pass
    
    def send_shards(self, rank_list=[]):
        '''
        Shard and send tensor to the specified rank_list for paralellization.
        '''
        pass
    
    def multiply(self, a, b):
        '''
        Multiply the sharded tensors.
        '''
        pass
    
    def add(self, a, b): # TODO: Check if this overrides the + operator
        '''
        Add the sharded tensors.
        '''
        pass