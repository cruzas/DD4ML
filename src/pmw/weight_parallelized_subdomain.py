import copy
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch import autograd, nn

import src.utils as utils
from src.pmw.base_model import BaseModel
from src.pmw.sharded_layer import ShardedLayer


class WeightParallelizedSubdomain(BaseModel):
    # This corresponds to a STAGE in a pipelined model.
    def __init__(self, model_handler):
        super().__init__()
        self.model_handler = model_handler
        self.sd, self.rep, self.s, self.sh = self.model_handler.sd, self.model_handler.rep, self.model_handler.s, self.model_handler.sh
        self.setup_phase = False

        self.inputs = {}
        self.outputs = {}
        self.grad_outputs = {}
        self.shapes = {}
        self.backward_shapes = {}

        self.stage_data = model_handler.stage_data()
        self.consec_layers = model_handler.get_list_of_consecutive_layers()
        self.sharded_layers = []
        self.connector_symbol = '|~|~|'
        # make sure that in the model_handler there is no connector_symbol in the layer names
        if self.DEBUG:
            print(
                f'(WeightParallelizedSubdomain INIT) rank={self.rank}) Layer order: {self.stage_data["layers"]}')
        for layer_name in self.stage_data['layers']:
            if self.connector_symbol in layer_name:
                raise ValueError(
                    f"Layer name {layer_name} contains the connector symbol {self.connector_symbol}. This is not allowed.")

        if self.rank in self.stage_data['ranks']:
            for layer_name in self.stage_data['layers']:
                self.sharded_layers.append(ShardedLayer(
                    layer_dict=self.model_handler.net_dict[layer_name], layer_ranks=self.stage_data['ranks']))

    def load_state_dict(self, state_dict):
        for layer_name in self.stage_data['layers']:
            self.sharded_layers[self.stage_data['layers'].index(
                layer_name)].load_state_dict(state_dict[layer_name])

    def state_dict(self):
        ordered_dict = OrderedDict()
        temp = {layer_name: layer.state_dict() for layer_name, layer in zip(
            self.stage_data['layers'], self.sharded_layers)}
        for key in temp.keys():
            for subkey in temp[key].keys():
                subkey2 = '.'.join(subkey.split('.')[1:])
                ordered_dict[key+'.'+subkey2] = temp[key][subkey]
        return ordered_dict

    def parameters(self):
        return [param for layer in self.sharded_layers for param in layer.parameters()]

    def forward(self, x=None, num_chunks=None, num_samples_in_chunk=None, chunk_id=None, is_in_pipeline=False):
        self.DEBUG = False
        empty_at_the_end = []
        if not is_in_pipeline:
            for chunk_id in range(len(self.outputs[list(self.outputs.keys())[0]])):
                for i, layer_name in enumerate(self.stage_data['layers']):
                    input_list = self.model_handler.net_dict[layer_name]['rcv']['src']
                    strategy_in = self.model_handler.net_dict[layer_name]['rcv']['strategy']
                    if strategy_in is None:
                        input_name = 'start' if layer_name == 'start' else input_list[0]
                        x = self.inputs[layer_name +
                                        self.connector_symbol+input_name][chunk_id]
                    else:
                        x = strategy_in(
                            *[self.inputs[layer_name+self.connector_symbol+src_name][chunk_id] for src_name in input_list])
                    # Forward pass. Output may be a list in case of multiple outputs (each output goes to a different destination layer)
                    x = self.sharded_layers[i].forward(x)

                    if layer_name == 'finish':
                        if chunk_id == 0:
                            self.outputs['finish'] = [None] * \
                                len(self.outputs['finish'])
                        self.outputs['finish'][chunk_id] = x

                    for dst_idx, dst_name in enumerate(self.model_handler.net_dict[layer_name]['dst']['to']):
                        key = layer_name+self.connector_symbol+dst_name
                        current_layer_stage = self.model_handler.net_dict[layer_name]['stage']
                        dst_layer_stage = self.model_handler.net_dict[dst_name]['stage']
                        temp = x if not isinstance(x, list) else x[dst_idx]
                        if current_layer_stage != dst_layer_stage:
                            if chunk_id == 0:
                                self.outputs[key] = [None] * \
                                    len(self.outputs[key])
                            self.outputs[key][chunk_id] = temp
                        else:
                            # Key: receiver is first, followed by the sender
                            reverse_key = dst_name+self.connector_symbol+layer_name
                            empty_at_the_end.append(reverse_key)
                            self.inputs[reverse_key] = [None] * \
                                len(self.inputs[reverse_key])
                            self.inputs[reverse_key][chunk_id] = temp

        else:  # Here we are in a pipeline and x is not None just for the first stage
            if self.DEBUG:
                print(
                    f'(FWD rank={self.rank}) Layer order: {self.stage_data["layers"]}')
            for i, layer_name in enumerate(self.stage_data['layers']):
                if layer_name == 'start':
                    if chunk_id == 0:
                        self.inputs['start'+self.connector_symbol +
                                    'start'] = [None]*num_chunks
                    self.inputs['start'+self.connector_symbol +
                                'start'][chunk_id] = x
                for src_name in self.model_handler.net_dict[layer_name]['fwd_rcv']['src']:
                    current_layer_stage = self.model_handler.net_dict[layer_name]['stage']
                    src_layer_stage = self.model_handler.net_dict[src_name]['stage']
                    key = layer_name+self.connector_symbol+src_name
                    if current_layer_stage != src_layer_stage:
                        src_ranks = self.model_handler.layer_name_to_ranks(
                            src_name)
                        # TODO: maybe improve send/rcv when tensor sharding is implemented
                        src_rank = src_ranks[0]
                        if self.setup_phase:
                            if self.DEBUG:
                                print(
                                    f'(FWD rank={self.rank}) Layer {layer_name} waiting to receive from rank {src_rank} a tensor')
                            rcv_shape = utils.receive_shape(
                                src=src_rank, device=self.backend_device())
                            self.shapes[key] = lambda z, temp_shape=copy.deepcopy(list(rcv_shape)[1:]): [
                                z] + temp_shape
                            if self.DEBUG:
                                print(
                                    f'(FWD rank={self.rank}) Layer {layer_name} received a tensor from rank {src_rank} with shape {rcv_shape}')
                        temp = torch.empty(self.shapes[key](
                            num_samples_in_chunk), device=self.backend_device(), requires_grad=True)
                        # TODO: make it async to speed up communication
                        dist.recv(src=src_rank, tensor=temp)
                        if chunk_id == 0 or key not in self.inputs.keys() or len(self.inputs[key]) != num_chunks:
                            self.inputs[key] = [None]*num_chunks
                        self.inputs[key][chunk_id] = temp.to(
                            self.tensor_device)

                input_list = self.model_handler.net_dict[layer_name]['rcv']['src']
                strategy_in = self.model_handler.net_dict[layer_name]['rcv']['strategy']
                if strategy_in is None:
                    input_name = 'start' if layer_name == 'start' else input_list[0]
                    x = self.inputs[layer_name +
                                    self.connector_symbol+input_name][chunk_id]
                else:
                    x = strategy_in(
                        *[self.inputs[layer_name+self.connector_symbol+src_name][chunk_id] for src_name in input_list])
                try:
                    out = self.sharded_layers[i].forward(x)
                except Exception as e:
                    if strategy_in is None:
                        raise ValueError(
                            f"Error {e} in layer {layer_name} during the forward pass.")
                    else:
                        raise ValueError(
                            f"Error {e} in layer {layer_name} during the forward pass. Are you sure that the strategy function takes inputs in the correct order?")

                if isinstance(out, list) and len(self.model_handler.net_dict[layer_name]['fwd_dst']['to']) != len(out):
                    raise ValueError(
                        f"Output of layer {layer_name} is a list of torch.Tensor with length different from the number of destination layers")
                elif not isinstance(out, torch.Tensor) and not isinstance(out, list):
                    raise TypeError(
                        f"Output of the callable object with label {layer_name} is of type {type(out)}. Only torch.Tensor or List (of torch.Tensor) is allowed.")

                if layer_name == 'finish':
                    if chunk_id == 0:
                        self.outputs['finish'] = [None]*num_chunks
                    self.outputs['finish'][chunk_id] = out.to(
                        self.tensor_device)

                for dst_idx, dst_name in enumerate(self.model_handler.net_dict[layer_name]['fwd_dst']['to']):
                    dst_ranks = self.model_handler.layer_name_to_ranks(
                        dst_name)
                    # TODO: maybe improve send/rcv when tensor sharding is implemented
                    dst_rank = dst_ranks[0]
                    key = layer_name+self.connector_symbol+dst_name
                    current_layer_stage = self.model_handler.net_dict[layer_name]['stage']
                    dst_layer_stage = self.model_handler.net_dict[dst_name]['stage']
                    temp = out if not isinstance(out, list) else out[dst_idx]
                    if current_layer_stage != dst_layer_stage:
                        temp = temp.to(self.backend_device())
                        if self.setup_phase:
                            if self.DEBUG:
                                print(
                                    f'(FWD rank={self.rank}) Layer {layer_name} sending to rank {dst_rank} a tensor with shape: {temp.shape}')
                            utils.send_shape(
                                shape=temp.shape, dst=dst_rank, device=self.backend_device())
                            if self.DEBUG:
                                print(
                                    f'(FWD rank={self.rank}) Layer {layer_name} sent a tensor to rank {dst_rank}')
                        dist.send(tensor=temp, dst=dst_rank)

                        if chunk_id == 0 or key not in self.outputs.keys() or len(self.outputs[key]) != num_chunks:
                            self.outputs[key] = [None]*num_chunks
                        self.outputs[key][chunk_id] = temp.to(
                            self.tensor_device)
                    else:
                        # Key: receiver is first, followed by the sender
                        reverse_key = dst_name+self.connector_symbol+layer_name
                        empty_at_the_end.append(reverse_key)
                        self.inputs[reverse_key] = [None]*num_chunks
                        self.inputs[reverse_key][chunk_id] = temp

        for key in empty_at_the_end:
            num_chunks = len(self.inputs[key])
            del self.inputs[key]
            self.inputs[key] = [None]*num_chunks

        if self.DEBUG:
            print(
                f'(FWD rank={self.rank}) Layer {layer_name} finished forward pass')
        return self.outputs['finish'] if self.model_handler.is_last_stage() else [True]

    def backward(self, loss=None, chunk_id=0, is_in_pipeline=False):
        # Not in a pipeline - subdomain independent backward (no communication)
        if not is_in_pipeline:
            for chunk_id in range(len(self.outputs[list(self.outputs.keys())[0]])):
                if self.model_handler.is_last_stage():  # End of the pipeline
                    loss_ = loss[chunk_id]
                    loss_.backward(retain_graph=True)
                    for name, inputs in self.inputs.items():
                        _, rcv_name = name.split(self.connector_symbol)
                        rcv_ranks = self.model_handler.layer_name_to_ranks(
                            rcv_name)
                        assert len(
                            rcv_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                        if self.rank != rcv_ranks[0]:
                            reverse_name = self.connector_symbol.join(
                                reversed(name.split(self.connector_symbol)))
                            if chunk_id == 0:
                                self.grad_outputs[reverse_name] = [
                                    None]*len(inputs)
                            self.grad_outputs[reverse_name][chunk_id] = torch.autograd.grad(
                                outputs=loss_, inputs=inputs[chunk_id], retain_graph=True)[0]
                else:
                    for name, outputs in self.outputs.items():
                        _, rcv_name = name.split(self.connector_symbol)
                        rcv_ranks = self.model_handler.layer_name_to_ranks(
                            rcv_name)
                        assert len(
                            rcv_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                        if self.rank != rcv_ranks[0] and outputs[chunk_id].requires_grad:
                            outputs[chunk_id].backward(
                                self.grad_outputs[name][chunk_id], retain_graph=True)
        else:
            for i, consecutive_block in enumerate(reversed(self.consec_layers)):
                bottom = True if 'finish' in consecutive_block else False
                if bottom:  # End of the pipeline
                    if i == 0:
                        loss.backward(retain_graph=True)
                    for current_layer in reversed(consecutive_block):
                        dst_names = [
                            name for name in self.model_handler.net_dict[current_layer]['bwd_dst']['to']]
                        for dst_name in dst_names:
                            dst_ranks = self.model_handler.layer_name_to_ranks(
                                dst_name)
                            assert len(
                                dst_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                            # Check if this block of consecutive layers is linked to "name"
                            # I.e. check that an external layer is actually connected to a layer in our current block
                            # (NOTE: this is to allow to have independent non consecutive blocks of layers on the same stage)
                            if self.rank != dst_ranks[0] and any([element in current_layer for element in consecutive_block]):
                                inputs = self.inputs[current_layer +
                                                     self.connector_symbol + dst_name]
                                reverse_name = dst_name + self.connector_symbol + current_layer
                                if chunk_id == 0:
                                    self.grad_outputs[reverse_name] = [
                                        None]*len(inputs)
                                self.grad_outputs[reverse_name][chunk_id] = torch.autograd.grad(
                                    outputs=loss, inputs=inputs[chunk_id], retain_graph=True)[0]
                                if self.setup_phase:
                                    if self.DEBUG:
                                        print(
                                            f'(BWD rank={self.rank}) Layer {current_layer} sending to rank {dst_ranks[0]} shape: {self.grad_outputs[reverse_name][chunk_id].shape}')
                                    utils.send_shape(
                                        shape=self.grad_outputs[reverse_name][chunk_id].shape, dst=dst_ranks[0], device=self.backend_device())
                                    if self.DEBUG:
                                        print(
                                            f'(BWD rank={self.rank}) Layer {current_layer} sent to rank {dst_ranks[0]}')
                                dist.send(tensor=self.grad_outputs[reverse_name][chunk_id].to(
                                    self.backend_device()), dst=dst_ranks[0])
                else:
                    for current_layer in reversed(consecutive_block):
                        rcv_names = [
                            name for name in self.model_handler.net_dict[current_layer]['bwd_rcv']['src']]
                        for rcv_name in rcv_names:
                            key = current_layer + self.connector_symbol + rcv_name
                            if any([element in current_layer for element in consecutive_block]):
                                rcv_ranks = self.model_handler.layer_name_to_ranks(
                                    rcv_name)
                                assert len(
                                    rcv_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                                if self.rank != rcv_ranks[0]:
                                    outputs = self.outputs[key]
                                    if self.setup_phase:
                                        if self.DEBUG:
                                            print(
                                                f'(BWD rank={self.rank}) Layer {current_layer} waiting to receive from rank {rcv_ranks[0]} the shape')
                                        rcv_shape = utils.receive_shape(
                                            src=rcv_ranks[0], device=self.backend_device())
                                        if self.DEBUG:
                                            print(
                                                f'(BWD rank={self.rank}) Layer {current_layer} received from rank {rcv_ranks[0]} shape {rcv_shape}')
                                        self.backward_shapes[key] = lambda z, temp_shape=copy.deepcopy(
                                            list(rcv_shape)[1:]): [z] + temp_shape
                                    grad_output = torch.empty(self.backward_shapes[key](
                                        outputs[chunk_id].shape[0]), device=self.backend_device(), requires_grad=True)
                                    dist.recv(tensor=grad_output,
                                              src=rcv_ranks[0])
                                    grad_output = grad_output.to(
                                        self.tensor_device).detach()
                                    if chunk_id == 0 or key not in self.grad_outputs.keys() or len(self.grad_outputs[key]) != len(outputs):
                                        self.grad_outputs[key] = [
                                            None]*len(outputs)
                                    self.grad_outputs[key][chunk_id] = grad_output
                                    if outputs[chunk_id].requires_grad:
                                        outputs[chunk_id].backward(
                                            grad_output, retain_graph=True)

                    # Collect all outputs and all gradients into one tensor each
                    all_outputs = [outputs[chunk_id] for key, outputs in self.outputs.items(
                    ) if any([element in key for element in consecutive_block])]
                    all_grads = [self.grad_outputs[key][chunk_id] for key in self.outputs.keys(
                    ) if any([element in key for element in consecutive_block])]
                    for current_layer in reversed(consecutive_block):
                        dst_names = [
                            name for name in self.model_handler.net_dict[current_layer]['bwd_dst']['to']]
                        for dst_name in dst_names:

                            dst_ranks = self.model_handler.layer_name_to_ranks(
                                dst_name)
                            assert len(
                                dst_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                            if self.rank != dst_ranks[0] and any([element in current_layer for element in consecutive_block]):
                                inputs = self.inputs[current_layer +
                                                     self.connector_symbol + dst_name]
                                grad_output = torch.autograd.grad(
                                    outputs=all_outputs, inputs=inputs[chunk_id], grad_outputs=all_grads, retain_graph=True)[0]
                                if self.setup_phase:
                                    if self.DEBUG:
                                        print(
                                            f'(BWD rank={self.rank}) Layer {current_layer} sending to rank {dst_ranks[0]} shape: {grad_output.shape}')
                                    utils.send_shape(
                                        shape=grad_output.shape, dst=dst_ranks[0], device=self.backend_device())
                                    if self.DEBUG:
                                        print(
                                            f'(BWD rank={self.rank}) Layer {current_layer} sent shape to rank {dst_ranks[0]}')
                                dist.send(tensor=grad_output.contiguous().to(
                                    self.backend_device()), dst=dst_ranks[0])

    def grad(self):
        # TODO: Implement sharded_layers.parameters()
        return [param.grad for param in self.sharded_layers.parameters()]

    def grad_norm(self):
        return torch.norm(torch.cat([param.grad.flatten() for param in self.sharded_layers.parameters()], dim=0), p=2).item()
