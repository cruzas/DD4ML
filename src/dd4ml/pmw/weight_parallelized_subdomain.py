import copy
from collections import OrderedDict
import time

import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

import dd4ml.utility as utils

_RPC_SUBDOMAIN = None


def register_rpc_subdomain(subdomain):
    """Register the current subdomain for RPC helpers."""
    global _RPC_SUBDOMAIN
    _RPC_SUBDOMAIN = subdomain


def _rpc_store_input(key, chunk_id, tensor, num_chunks):
    """Store a received tensor in the inputs dictionary."""
    sd = _RPC_SUBDOMAIN
    sd._ensure_list(sd.inputs, key, num_chunks)
    sd.inputs[key][chunk_id] = tensor.to(sd.tensor_device)


def _rpc_store_grad(key, chunk_id, tensor, num_chunks):
    """Store a received tensor in the grad_outputs dictionary."""
    sd = _RPC_SUBDOMAIN
    sd._ensure_list(sd.grad_outputs, key, num_chunks)
    sd.grad_outputs[key][chunk_id] = tensor.to(sd.tensor_device)


def _rpc_get_output(key, chunk_id):
    """Retrieve an output tensor waiting until it's available."""
    sd = _RPC_SUBDOMAIN
    while key not in sd.outputs or sd.outputs[key][chunk_id] is None:
        time.sleep(0.001)
    return sd.outputs[key][chunk_id].cpu()


def _rpc_get_grad(key, chunk_id):
    """Retrieve a gradient tensor waiting until it's available."""
    sd = _RPC_SUBDOMAIN
    while key not in sd.grad_outputs or sd.grad_outputs[key][chunk_id] is None:
        time.sleep(0.001)
    return sd.grad_outputs[key][chunk_id].cpu()

from .base_pmw_model import BasePMWModel
from .sharded_layer import ShardedLayer


class WeightParallelizedSubdomain(BasePMWModel):
    def __init__(self, model_handler):
        super().__init__()
        self.model_handler = model_handler
        self.sd, self.rep, self.s, self.sh = (
            model_handler.sd,
            model_handler.rep,
            model_handler.s,
            model_handler.sh,
        )
        self.setup_phase = False
        self.inputs, self.outputs, self.grad_outputs = {}, {}, {}
        self.shapes, self.backward_shapes = {}, {}
        self.stage_data = model_handler.stage_data()
        self.consec_layers = model_handler.get_list_of_consecutive_layers()
        self.connector_symbol = "|~|~|"
        if self.DEBUG:
            print(f'(INIT) rank={self.rank} Layer order: {self.stage_data["layers"]}')
        if any(self.connector_symbol in name for name in self.stage_data["layers"]):
            raise ValueError(f"Layer names cannot contain the connector symbol {self.connector_symbol}.")
        self.sharded_layers = (
            [ShardedLayer(layer_dict=model_handler.net_dict[name],
                          layer_ranks=self.stage_data["ranks"])
             for name in self.stage_data["layers"]]
            if self.rank in self.stage_data["ranks"] else []
        )

        if not rpc.is_initialized():
            options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16)
            rpc.init_rpc(
                name=f"worker{self.rank}",
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=options,
            )
        register_rpc_subdomain(self)

    def _key(self, a, b):
        return f"{a}{self.connector_symbol}{b}"

    def _ensure_list(self, d, key, size):
        if key not in d or len(d[key]) != size:
            d[key] = [None] * size
        return d[key]

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
        return [p for layer in self.sharded_layers for p in layer.parameters()]

    def named_parameters(self):
        return [(name, p) for name, layer in zip(self.stage_data["layers"], self.sharded_layers)
                for p in layer.parameters()]

    def configure_params(self, train_config):
        return [layer.configure_params(train_config) for layer in self.sharded_layers]

    def forward(self, x=None, num_chunks=None, num_samples_in_chunk=None, chunk_id=None, is_in_pipeline=False):
        if not is_in_pipeline:
            return self._forward_non_pipeline()
        return self._forward_pipeline(x, num_chunks, num_samples_in_chunk, chunk_id)

    def _forward_non_pipeline(self):
        self.DEBUG = False
        empty_at_the_end = []
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
                        *[
                            self.inputs[self._key(layer_name, src)][chunk]
                            for src in input_list
                        ]
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
                        self._ensure_list(
                            self.inputs, reverse_key, num_chunks_local
                        )
                        self.inputs[reverse_key][chunk] = temp
        for key in empty_at_the_end:
            num_chunks_local = len(self.inputs[key])
            del self.inputs[key]
            self.inputs[key] = [None] * num_chunks_local
        return (
            self.outputs["finish"] if self.model_handler.is_last_stage() else [True]
        )
    
    def _forward_pipeline(self, x=None, num_chunks=None, num_samples_in_chunk=None, chunk_id=None, ):
        self.DEBUG = False
        empty_at_the_end = []
        
        backend_dev = self.backend_device()
        tensor_dev = self.tensor_device
        for i, layer_name in enumerate(self.stage_data["layers"]):
            net_dict = self.model_handler.net_dict[layer_name]
            if layer_name == "start":
                self._ensure_list(
                    self.inputs, self._key("start", "start"), num_chunks
                )
                self.inputs[self._key("start", "start")][chunk_id] = x
            for src_name in net_dict["fwd_rcv"]["src"]:
                key = self._key(layer_name, src_name)
                current_layer_stage = net_dict["stage"]
                src_layer_stage = self.model_handler.net_dict[src_name]["stage"]
                if current_layer_stage != src_layer_stage:
                    src_ranks = self.model_handler.layer_name_to_ranks(src_name)
                    src_rank = src_ranks[0]
                    temp = rpc.rpc_sync(
                        to=f"worker{src_rank}",
                        func=_rpc_get_output,
                        args=(self._key(src_name, layer_name), chunk_id),
                    )
                    temp = temp.to(tensor_dev).requires_grad_()
                    self._ensure_list(self.inputs, key, num_chunks)
                    self.inputs[key][chunk_id] = temp
            if net_dict["rcv"]["strategy"] is None:
                input_name = (
                    "start" if layer_name == "start" else net_dict["rcv"]["src"][0]
                )
                x = self.inputs[self._key(layer_name, input_name)][chunk_id]
            else:
                x = net_dict["rcv"]["strategy"](
                    *[
                        self.inputs[self._key(layer_name, src)][chunk_id]
                        for src in net_dict["rcv"]["src"]
                    ]
                )
            try:
                out = self.sharded_layers[i].forward(x)
            except Exception as e:
                if net_dict["rcv"]["strategy"] is None:
                    raise ValueError(
                        f"Error {e} in layer {layer_name} during the forward pass."
                    )
                else:
                    raise ValueError(
                        f"Error {e} in layer {layer_name} during the forward pass. Are you sure that the strategy function takes inputs in the correct order?"
                    )
            if isinstance(out, list) and len(net_dict["fwd_dst"]["to"]) != len(out):
                raise ValueError(
                    f"Output of layer {layer_name} is a list of torch.Tensor with length different from the number of destination layers"
                )
            elif not isinstance(out, torch.Tensor) and not isinstance(out, list):
                raise TypeError(
                    f"Output of the callable object with label {layer_name} is of type {type(out)}. Only torch.Tensor or List (of torch.Tensor) is allowed."
                )
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
                    temp_backend = temp.to(backend_dev)
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
        return (
            self.outputs["finish"] if self.model_handler.is_last_stage() else [True]
        )    
    
    def backward(self, loss=None, chunk_id=0, is_in_pipeline=False):
        if not is_in_pipeline:
            self._backward_non_pipeline(loss)
        else:
            self._backward_pipeline(loss, chunk_id)

    def _backward_non_pipeline(self, loss):
        num_chunks_local = len(self.outputs[next(iter(self.outputs))])
        for chunk in range(num_chunks_local):
            if self.model_handler.is_last_stage():
                loss_ = loss[chunk]
                loss_.backward(retain_graph=True)
                for name, inputs in self.inputs.items():
                    _, rcv_name = name.split(self.connector_symbol)
                    rcv_ranks = self.model_handler.layer_name_to_ranks(rcv_name)
                    assert (
                        len(rcv_ranks) == 1
                    ), "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                    if self.rank != rcv_ranks[0]:
                        reverse_name = self.connector_symbol.join(
                            reversed(name.split(self.connector_symbol))
                        )
                        self._ensure_list(
                            self.grad_outputs, reverse_name, len(inputs)
                        )
                        self.grad_outputs[reverse_name][chunk] = (
                            torch.autograd.grad(
                                outputs=loss_,
                                inputs=inputs[chunk],
                                retain_graph=True,
                            )[0]
                        )
            else:
                for name, outputs in self.outputs.items():
                    _, rcv_name = name.split(self.connector_symbol)
                    rcv_ranks = self.model_handler.layer_name_to_ranks(rcv_name)
                    assert (
                        len(rcv_ranks) == 1
                    ), "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                    if self.rank != rcv_ranks[0] and outputs[chunk].requires_grad:
                        outputs[chunk].backward(
                            self.grad_outputs[name][chunk], retain_graph=True
                        )
                        
    def _backward_pipeline(self, loss, chunk_id):
        backend_dev = self.backend_device()
        tensor_dev = self.tensor_device
        for i, consecutive_block in enumerate(reversed(self.consec_layers)):
            bottom = "finish" in consecutive_block
            if bottom:
                if i == 0:
                    loss.backward(retain_graph=True)
                for current_layer in reversed(consecutive_block):
                    dst_names = self.model_handler.net_dict[current_layer][
                        "bwd_dst"
                    ]["to"]
                    for dst_name in dst_names:
                        dst_ranks = self.model_handler.layer_name_to_ranks(dst_name)
                        assert (
                            len(dst_ranks) == 1
                        ), "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                        if self.rank != dst_ranks[0] and any(
                            [
                                element in current_layer
                                for element in consecutive_block
                            ]
                        ):
                            inputs = self.inputs[self._key(current_layer, dst_name)]
                            reverse_name = self._key(dst_name, current_layer)
                            self._ensure_list(
                                self.grad_outputs, reverse_name, len(inputs)
                            )
                            self.grad_outputs[reverse_name][chunk_id] = (
                                torch.autograd.grad(
                                    outputs=loss,
                                    inputs=inputs[chunk_id],
                                    retain_graph=True,
                                )[0]
                            )
                            rpc.rpc_sync(
                                to=f"worker{dst_ranks[0]}",
                                func=_rpc_store_grad,
                                args=(
                                    self._key(dst_name, current_layer),
                                    chunk_id,
                                    self.grad_outputs[reverse_name][chunk_id].to(backend_dev),
                                    len(inputs),
                                ),
                            )
            else:
                for current_layer in reversed(consecutive_block):
                    rcv_names = self.model_handler.net_dict[current_layer][
                        "bwd_rcv"
                    ]["src"]
                    for rcv_name in rcv_names:
                        key = self._key(current_layer, rcv_name)
                        if any(
                            [
                                element in current_layer
                                for element in consecutive_block
                            ]
                        ):
                            rcv_ranks = self.model_handler.layer_name_to_ranks(
                                rcv_name
                            )
                            assert (
                                len(rcv_ranks) == 1
                            ), "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                            if self.rank != rcv_ranks[0]:
                                outputs = self.outputs[key]
                                grad_output = rpc.rpc_sync(
                                    to=f"worker{rcv_ranks[0]}",
                                    func=_rpc_get_grad,
                                    args=(self._key(rcv_name, current_layer), chunk_id),
                                )
                                grad_output = grad_output.to(tensor_dev).detach()
                                self._ensure_list(
                                    self.grad_outputs, key, len(outputs)
                                )
                                self.grad_outputs[key][chunk_id] = grad_output
                                if outputs[chunk_id].requires_grad:
                                    outputs[chunk_id].backward(
                                        grad_output, retain_graph=True
                                    )
                all_outputs = [
                    outputs[chunk_id]
                    for key, outputs in self.outputs.items()
                    if any([element in key for element in consecutive_block])
                ]
                all_grads = [
                    self.grad_outputs[key][chunk_id]
                    for key in self.outputs.keys()
                    if any([element in key for element in consecutive_block])
                ]
                for current_layer in reversed(consecutive_block):
                    dst_names = self.model_handler.net_dict[current_layer][
                        "bwd_dst"
                    ]["to"]
                    for dst_name in dst_names:
                        dst_ranks = self.model_handler.layer_name_to_ranks(dst_name)
                        assert (
                            len(dst_ranks) == 1
                        ), "Tensor sharding not implemented yet. Only one rank per layer is supported for now"
                        if self.rank != dst_ranks[0] and any(
                            [
                                element in current_layer
                                for element in consecutive_block
                            ]
                        ):
                            inputs = self.inputs[self._key(current_layer, dst_name)]
                            grad_output = torch.autograd.grad(
                                outputs=all_outputs,
                                inputs=inputs[chunk_id],
                                grad_outputs=all_grads,
                                retain_graph=True,
                            )[0]
                            rpc.rpc_sync(
                                to=f"worker{dst_ranks[0]}",
                                func=_rpc_store_grad,
                                args=(self._key(dst_name, current_layer), chunk_id, grad_output.contiguous().to(backend_dev), len(inputs)),
                            )
                            
    def grad(self):
        return [p.grad for p in self.sharded_layers.parameters()]

    def grad_norm(self):
        return torch.norm(
            torch.cat(
                [p.grad.flatten() for p in self.sharded_layers.parameters()],
                dim=0,
            ),
            p=2,
        )
