import time

import torch
import torch.autograd as autograd
import torch.distributed as dist
import torch.nn as nn

from .base_pmw_model import BasePMWModel
from .weight_parallelized_subdomain import WeightParallelizedSubdomain
from .weight_parallelized_tensor import WeightParallelizedTensor


class WeightParallelizedModel(BasePMWModel):
    def __init__(self, model_handler, sample):
        """
        NOTE: grad_norm function returns the infinity norm of the subdomain gradient of the model (i.e. restricted to the current rank).
        Assumptions:
        1) Each sample has shape[0] = batch size -> each row is a sample.
        2) Only one layer is passed in the stage_list. Note that this counts sequential layers as one layer (e.g. nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,300)) counts as one layer).
        """
        super().__init__()
        self.subdomain = WeightParallelizedSubdomain(
            model_handler
        )  # Initialize the subdomain model
        self.model_handler = model_handler
        self.first_stage_ranks = self.model_handler.get_stage_ranks(
            stage_name="first", mode="replica"
        )
        self.do_setup_phase(sample)

    def do_setup_phase(self, sample):
        loss = None
        self.subdomain.setup_phase = True
        if sample.device != self.tensor_device:
            sample = sample.to(self.tensor_device)
        out = self.forward(sample, chunks_amount=1, reset_grad=True, compute_grad=True)
        if self.model_handler.is_last_stage():
            loss = nn.MSELoss()(out[0], torch.rand_like(out[0]))
        self.backward(losses=[loss])
        self.zero_grad()
        self.subdomain.setup_phase = False

    def zero_grad(self):
        self.subdomain.zero_grad()

    def grad(self, clone=True):  # Returns the global gradient of the model
        gradient = [
            param.grad.clone() if clone else param.grad for param in self.parameters()
        ]
        return WeightParallelizedTensor(
            gradient, self.backend, self.model_handler.get_replica_group(), self.rank
        )

    def grad_norm(self, p=2):  # Returns the global gradient norm of the model
        return self.grad().norm(p=p)

    def parameters(self, clone=True):  # Returns the global parameters of the model
        params = [
            param.clone() if clone else param for param in self.subdomain.parameters()
        ]
        return WeightParallelizedTensor(
            params, self.backend, self.model_handler.get_replica_group(), self.rank
        )

    # Returns the subdomain gradient norm of the model
    def subdomain_grad_norm(self, p=2):
        return torch.norm(
            torch.cat([param.grad.flatten() for param in self.parameters()], dim=0), p=p
        )

    def forward(self, x, chunks_amount=1, reset_grad=False, compute_grad=True):
        # flag to avoid storing the tensors needed to compute the gradients
        compute_grad = False if not torch.is_grad_enabled() else compute_grad
        with torch.set_grad_enabled(compute_grad):
            if reset_grad:
                self.zero_grad()  # Reset the gradients of the model before starting to accumulate them again

            # Initialize the chunk_shapes tensor to store the shapes of the chunks
            chunk_shapes = torch.zeros(
                chunks_amount, dtype=torch.int32, device=self.backend_device(x)
            )

            # If the rank is in the first stage, process the input
            if self.model_handler.is_first_stage():
                # Chunkenize the input tensor
                chunks = x.chunk(
                    chunks_amount
                )  # Direct tuple unpacking instead of list()

                # Store batch sizes efficiently
                chunk_shapes[: len(chunks)] = torch.tensor(
                    [chunk.shape[0] for chunk in chunks],
                    dtype=torch.int32,
                    device=chunk_shapes.device,
                )

            # Broadcast chunk_shapes to all ranks
            dist.broadcast(
                tensor=chunk_shapes,
                src=self.first_stage_ranks[0],
                group=self.model_handler.get_replica_group(),
                async_op=False,
            )

            # Iterate through pipeline
            for c in range(chunks_amount):
                temp = (
                    chunks[c].to(self.tensor_device)
                    if self.model_handler.is_first_stage()
                    else None
                )
                self.subdomain.forward(
                    num_chunks=chunks_amount,
                    num_samples_in_chunk=chunk_shapes[c].item(),
                    chunk_id=c,
                    x=temp,
                    is_in_pipeline=True,
                )

        if self.model_handler.is_last_stage():
            return self.subdomain.outputs["finish"]
        else:
            return [True]

        # return self.subdomain.outputs['finish'] if self.model_handler.is_last_stage() else [True]

    def backward(self, losses):
        num_chunks = len(losses)
        for i, loss in enumerate(losses):  # Chunked loss
            self.subdomain.backward(loss, chunk_id=i, is_in_pipeline=True)

        # Rescale the gradients by the number of chunks
        for param in self.parameters():
            param.grad /= num_chunks
