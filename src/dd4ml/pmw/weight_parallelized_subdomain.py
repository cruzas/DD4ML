import copy
from collections import OrderedDict

import torch
import torch.distributed as dist

import dd4ml.utils as utils
from dd4ml.pmw.base_pmw_model import BasePMWModel
from dd4ml.pmw.sharded_layer import ShardedLayer


class WeightParallelizedSubdomain(BasePMWModel):
    def __init__(self, model_handler):
        super().__init__()
        self.model_handler = model_handler
        self.sd, self.rep, self.s, self.sh = (
            self.model_handler.sd,
            self.model_handler.rep,
            self.model_handler.s,
            self.model_handler.sh,
        )
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
        if self.DEBUG:
            print(
                f'(WeightParallelizedSubdomain INIT) rank={self.rank}) Layer order: {self.stage_data["layers"]}'
            )
        for layer_name in self.stage_data["layers"]:
            if self.connector_symbol in layer_name:
                raise ValueError(
                    f"Layer name {layer_name} contains the connector symbol {self.connector_symbol}. This is not allowed."
                )
        if self.rank in self.stage_data["ranks"]:
            for layer_name in self.stage_data["layers"]:
                self.sharded_layers.append(
                    ShardedLayer(
                        layer_dict=self.model_handler.net_dict[layer_name],
                        layer_ranks=self.stage_data["ranks"],
                    )
                )

    def _key(self, a, b):
        return f"{a}{self.connector_symbol}{b}"

    def _ensure_list(self, dict_obj, key, size):
        if key not in dict_obj or len(dict_obj[key]) != size:
            dict_obj[key] = [None] * size
        return dict_obj[key]

    def load_state_dict(self, state_dict):
        for layer_name in self.stage_data["layers"]:
            idx = self.stage_data["layers"].index(layer_name)
            self.sharded_layers[idx].load_state_dict(state_dict[layer_name])

    def state_dict(self):
        ordered_dict = OrderedDict()
        temp = {
            layer_name: layer.state_dict()
            for layer_name, layer in zip(self.stage_data["layers"], self.sharded_layers)
        }
        for key in temp.keys():
            for subkey in temp[key].keys():
                subkey2 = ".".join(subkey.split(".")[1:])
                ordered_dict[f"{key}.{subkey2}"] = temp[key][subkey]
        return ordered_dict

    def parameters(self):
        return [param for layer in self.sharded_layers for param in layer.parameters()]

    def named_parameters(self):
        return [
            (layer_name, param)
            for layer_name, layer in zip(self.stage_data["layers"], self.sharded_layers)
            for param in layer.parameters()
        ]

    def configure_params(self, train_config):
        return [layer.configure_params(train_config) for layer in self.sharded_layers]

    def forward(self, x=None, num_chunks=None, num_samples_in_chunk=None, chunk_id=None, is_in_pipeline=False):
        self.DEBUG = False
        empty_at_the_end = []

        if not is_in_pipeline:
            num_chunks_local = len(self.outputs[next(iter(self.outputs))])
            for chunk in range(num_chunks_local):
                for i, layer_name in enumerate(self.stage_data["layers"]):
                    net_dict = self.model_handler.net_dict[layer_name]
                    input_list = net_dict["rcv"]["src"]
                    strategy_in = net_dict["rcv"]["strategy"]
                    if strategy_in is None:
                        input_name = "start" if layer_name == "start" else input_list[0]
                        x = self.inputs[self._key(layer_name, input_name)][chunk]
                    else:
                        x = strategy_in(
                            *[self.inputs[self._key(layer_name, src)][chunk] for src in input_list]
                        )
                    x = self.sharded_layers[i].forward(x)
                    if layer_name == "finish":
                        self._ensure_list(self.outputs, "finish", num_chunks_local)
                        self.outputs["finish"][chunk] = x
                    for dst_idx, dst_name in enumerate(net_dict["dst"]["to"]):
                        key = self._key(layer_name, dst_name)
                        current_layer_stage = net_dict["stage"]
                        dst_layer_stage = self.model_handler.net_dict[dst_name]["stage"]
                        temp = x if not isinstance(x, list) else x[dst_idx]
                        if current_layer_stage != dst_layer_stage:
                            self._ensure_list(self.outputs, key, num_chunks_local)
                            self.outputs[key][chunk] = temp
                        else:
                            reverse_key = self._key(dst_name, layer_name)
                            empty_at_the_end.append(reverse_key)
                            self._ensure_list(self.inputs, reverse_key, num_chunks_local)
                            self.inputs[reverse_key][chunk] = temp
            for key in empty_at_the_end:
                num_chunks_local = len(self.inputs[key])
                del self.inputs[key]
                self.inputs[key] = [None] * num_chunks_local
            return self.outputs["finish"] if self.model_handler.is_last_stage() else [True]

        else:
            backend_dev = self.backend_device()
            tensor_dev = self.tensor_device
            for i, layer_name in enumerate(self.stage_data["layers"]):
                net_dict = self.model_handler.net_dict[layer_name]
                if layer_name == "start":
                    self._ensure_list(self.inputs, self._key("start", "start"), num_chunks)
                    self.inputs[self._key("start", "start")][chunk_id] = x
                for src_name in net_dict["fwd_rcv"]["src"]:
                    key = self._key(layer_name, src_name)
                    current_layer_stage = net_dict["stage"]
                    src_layer_stage = self.model_handler.net_dict[src_name]["stage"]
                    if current_layer_stage != src_layer_stage:
                        src_ranks = self.model_handler.layer_name_to_ranks(src_name)
                        src_rank = src_ranks[0]
                        if self.setup_phase:
                            if self.DEBUG:
                                print(f'(FWD rank={self.rank}) Layer {layer_name} waiting to receive from rank {src_rank} a tensor')
                            rcv_shape = utils.receive_shape(src=src_rank, device=backend_dev)
                            self.shapes[key] = lambda z, temp_shape=copy.deepcopy(list(rcv_shape)[1:]): [z] + temp_shape
                            if self.DEBUG:
                                print(f'(FWD rank={self.rank}) Layer {layer_name} received a tensor from rank {src_rank} with shape {rcv_shape}')
                        temp = torch.empty(self.shapes[key](num_samples_in_chunk), device=backend_dev, requires_grad=True)
                        recv_handle = dist.irecv(tensor=temp, src=src_rank)
                        recv_handle.wait()
                        self._ensure_list(self.inputs, key, num_chunks)
                        self.inputs[key][chunk_id] = temp.to(tensor_dev)
                if net_dict["rcv"]["strategy"] is None:
                    input_name = "start" if layer_name == "start" else net_dict["rcv"]["src"][0]
                    x = self.inputs[self._key(layer_name, input_name)][chunk_id]
                else:
                    x = net_dict["rcv"]["strategy"](
                        *[self.inputs[self._key(layer_name, src)][chunk_id] for src in net_dict["rcv"]["src"]]
                    )
                try:
                    out = self.sharded_layers[i].forward(x)
                except Exception as e:
                    if net_dict["rcv"]["strategy"] is None:
                        raise ValueError(f"Error {e} in layer {layer_name} during the forward pass.")
                    else:
                        raise ValueError(
                            f"Error {e} in layer {layer_name} during the forward pass. Are you sure that the strategy function takes inputs in the correct order?"
                        )
                if isinstance(out, list) and len(net_dict["fwd_dst"]["to"]) != len(out):
                    raise ValueError(f"Output of layer {layer_name} is a list of torch.Tensor with length different from the number of destination layers")
                elif not isinstance(out, torch.Tensor) and not isinstance(out, list):
                    raise TypeError(f"Output of the callable object with label {layer_name} is of type {type(out)}. Only torch.Tensor or List (of torch.Tensor) is allowed.")
                if layer_name == "finish":
                    self._ensure_list(self.outputs, "finish", num_chunks)
                    self.outputs["finish"][chunk_id] = out.to(tensor_dev)
                for dst_idx, dst_name in enumerate(net_dict["fwd_dst"]["to"]):
                    dst_ranks = self.model_handler.layer_name_to_ranks(dst_name)
                    dst_rank = dst_ranks[0]
                    key = self._key(layer_name, dst_name)
                    current_layer_stage = net_dict["stage"]
                    dst_layer_stage = self.model_handler.net_dict[dst_name]["stage"]
                    temp = out if not isinstance(out, list) else out[dst_idx]
                    if current_layer_stage != dst_layer_stage:
                        temp = temp.to(backend_dev)
                        if self.setup_phase:
                            if self.DEBUG:
                                print(f'(FWD rank={self.rank}) Layer {layer_name} sending to rank {dst_rank} a tensor with shape: {temp.shape}')
                            utils.send_shape(shape=temp.shape, dst=dst_rank, device=backend_dev)
                            if self.DEBUG:
                                print(f'(FWD rank={self.rank}) Layer {layer_name} sent a tensor to rank {dst_rank}')
                        send_handle = dist.isend(tensor=temp, dst=dst_rank)
                        send_handle.wait()
                        self._ensure_list(self.outputs, key, num_chunks)
                        self.outputs[key][chunk_id] = temp.to(tensor_dev)
                    else:
                        reverse_key = self._key(dst_name, layer_name)
                        empty_at_the_end.append(reverse_key)
                        self._ensure_list(self.inputs, reverse_key, num_chunks)
                        self.inputs[reverse_key][chunk_id] = temp
            for key in empty_at_the_end:
                num_chunks_local = len(self.inputs[key])
                del self.inputs[key]
                self.inputs[key] = [None] * num_chunks_local
            return self.outputs["finish"] if self.model_handler.is_last_stage() else [True]

    def backward(self, loss=None, chunk_id=0, is_in_pipeline=False):
        if not is_in_pipeline:
            num_chunks_local = len(self.outputs[next(iter(self.outputs))])
            for chunk in range(num_chunks_local):
                if self.model_handler.is_last_stage():
                    loss_ = loss[chunk]
                    loss_.backward(retain_graph=True)
                    for name, inputs in self.inputs.items():
                        _, rcv_name = name.split(self.connector_symbol)
                        rcv_ranks = self.model_handler.layer_name_to_ranks(rcv_name)
                        assert len(rcv_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                        if self.rank != rcv_ranks[0]:
                            reverse_name = self.connector_symbol.join(reversed(name.split(self.connector_symbol)))
                            self._ensure_list(self.grad_outputs, reverse_name, len(inputs))
                            self.grad_outputs[reverse_name][chunk] = torch.autograd.grad(
                                outputs=loss_, inputs=inputs[chunk], retain_graph=True
                            )[0]
                else:
                    for name, outputs in self.outputs.items():
                        _, rcv_name = name.split(self.connector_symbol)
                        rcv_ranks = self.model_handler.layer_name_to_ranks(rcv_name)
                        assert len(rcv_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                        if self.rank != rcv_ranks[0] and outputs[chunk].requires_grad:
                            outputs[chunk].backward(self.grad_outputs[name][chunk], retain_graph=True)
        else:
            backend_dev = self.backend_device()
            tensor_dev = self.tensor_device
            for i, consecutive_block in enumerate(reversed(self.consec_layers)):
                bottom = "finish" in consecutive_block
                if bottom:
                    if i == 0:
                        loss.backward(retain_graph=True)
                    for current_layer in reversed(consecutive_block):
                        dst_names = self.model_handler.net_dict[current_layer]["bwd_dst"]["to"]
                        for dst_name in dst_names:
                            dst_ranks = self.model_handler.layer_name_to_ranks(dst_name)
                            assert len(dst_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                            if self.rank != dst_ranks[0] and any([element in current_layer for element in consecutive_block]):
                                inputs = self.inputs[self._key(current_layer, dst_name)]
                                reverse_name = self._key(dst_name, current_layer)
                                self._ensure_list(self.grad_outputs, reverse_name, len(inputs))
                                self.grad_outputs[reverse_name][chunk_id] = torch.autograd.grad(
                                    outputs=loss, inputs=inputs[chunk_id], retain_graph=True
                                )[0]
                                if self.setup_phase:
                                    if self.DEBUG:
                                        print(f'(BWD rank={self.rank}) Layer {current_layer} sending to rank {dst_ranks[0]} shape: {self.grad_outputs[reverse_name][chunk_id].shape}')
                                    utils.send_shape(shape=self.grad_outputs[reverse_name][chunk_id].shape, dst=dst_ranks[0], device=backend_dev)
                                    if self.DEBUG:
                                        print(f'(BWD rank={self.rank}) Layer {current_layer} sent to rank {dst_ranks[0]}')
                                send_handle = dist.isend(
                                    tensor=self.grad_outputs[reverse_name][chunk_id].to(backend_dev), dst=dst_ranks[0]
                                )
                                send_handle.wait()
                else:
                    for current_layer in reversed(consecutive_block):
                        rcv_names = self.model_handler.net_dict[current_layer]["bwd_rcv"]["src"]
                        for rcv_name in rcv_names:
                            key = self._key(current_layer, rcv_name)
                            if any([element in current_layer for element in consecutive_block]):
                                rcv_ranks = self.model_handler.layer_name_to_ranks(rcv_name)
                                assert len(rcv_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                                if self.rank != rcv_ranks[0]:
                                    outputs = self.outputs[key]
                                    if self.setup_phase:
                                        if self.DEBUG:
                                            print(f'(BWD rank={self.rank}) Layer {current_layer} waiting to receive from rank {rcv_ranks[0]} the shape')
                                        rcv_shape = utils.receive_shape(src=rcv_ranks[0], device=backend_dev)
                                        if self.DEBUG:
                                            print(f'(BWD rank={self.rank}) Layer {current_layer} received from rank {rcv_ranks[0]} shape {rcv_shape}')
                                        self.backward_shapes[key] = lambda z, temp_shape=copy.deepcopy(list(rcv_shape)[1:]): [z] + temp_shape
                                    grad_output = torch.empty(
                                        self.backward_shapes[key](outputs[chunk_id].shape[0]),
                                        device=backend_dev,
                                        requires_grad=True,
                                    )
                                    recv_handle = dist.irecv(tensor=grad_output, src=rcv_ranks[0])
                                    recv_handle.wait()
                                    grad_output = grad_output.to(tensor_dev).detach()
                                    self._ensure_list(self.grad_outputs, key, len(outputs))
                                    self.grad_outputs[key][chunk_id] = grad_output
                                    if outputs[chunk_id].requires_grad:
                                        outputs[chunk_id].backward(grad_output, retain_graph=True)
                    all_outputs = [outputs[chunk_id] for key, outputs in self.outputs.items() if any([element in key for element in consecutive_block])]
                    all_grads = [self.grad_outputs[key][chunk_id] for key in self.outputs.keys() if any([element in key for element in consecutive_block])]
                    for current_layer in reversed(consecutive_block):
                        dst_names = self.model_handler.net_dict[current_layer]["bwd_dst"]["to"]
                        for dst_name in dst_names:
                            dst_ranks = self.model_handler.layer_name_to_ranks(dst_name)
                            assert len(dst_ranks) == 1, "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                            if self.rank != dst_ranks[0] and any([element in current_layer for element in consecutive_block]):
                                inputs = self.inputs[self._key(current_layer, dst_name)]
                                grad_output = torch.autograd.grad(
                                    outputs=all_outputs,
                                    inputs=inputs[chunk_id],
                                    grad_outputs=all_grads,
                                    retain_graph=True,
                                )[0]
                                if self.setup_phase:
                                    if self.DEBUG:
                                        print(f'(BWD rank={self.rank}) Layer {current_layer} sending to rank {dst_ranks[0]} shape: {grad_output.shape}')
                                    utils.send_shape(shape=grad_output.shape, dst=dst_ranks[0], device=backend_dev)
                                    if self.DEBUG:
                                        print(f'(BWD rank={self.rank}) Layer {current_layer} sent shape to rank {dst_ranks[0]}')
                                send_handle = dist.isend(
                                    tensor=grad_output.contiguous().to(backend_dev), dst=dst_ranks[0]
                                )
                                send_handle.wait()

    def grad(self):
        return [param.grad for param in self.sharded_layers.parameters()]

    def grad_norm(self):
        return torch.norm(
            torch.cat([param.grad.flatten() for param in self.sharded_layers.parameters()], dim=0),
            p=2,
        ).item()
