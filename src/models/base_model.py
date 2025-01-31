import math
from abc import ABC, abstractmethod
from collections import deque

import torch.fx as fx
import torch.nn as nn

from src.utils import CfgNode as CN
from src.utils import is_function_module


class BaseModel(nn.Module, ABC):
    @staticmethod
    def get_default_config():
        C = CN()
        C.num_stages = 1
        C.num_subdomains = 1
        C.num_replicas_per_subdomain = 1
        return C
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_dict = None
    
    @staticmethod
    def set_stage(model_dict, num_stages):
        """
        Topologically sort modules, then assign them to pipeline stages.
        Consecutive 'function-like' modules (e.g. <function relu at 0x...>)
        are grouped into a single bundle so they share the same stage.
        """

        # 1. Build adjacency (forward edges) & in-degree
        adjacency = {}
        in_degree = {}
        for key, info in model_dict.items():
            adjacency[key] = info['dst']['to']
            in_degree[key] = 0

        for src, children in adjacency.items():
            for c in children:
                in_degree[c] += 1

        # 2. Topological sort (Kahn's Algorithm)
        queue = deque(k for k, deg in in_degree.items() if deg == 0)
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for child in adjacency[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(topo_order) < len(model_dict):
            raise ValueError("Cycle or unreachable nodes in the graph!")

        # 3. Merge consecutive function modules
        grouped = []
        current_group = [topo_order[0]]

        for i in range(1, len(topo_order)):
            prev_node = topo_order[i - 1]
            curr_node = topo_order[i]
            # Check if both are function modules and curr is the sole child of prev.
            if (is_function_module(model_dict[prev_node]) and
                is_function_module(model_dict[curr_node]) and
                adjacency[prev_node] == [curr_node]):
                current_group.append(curr_node)
            else:
                grouped.append(current_group)
                current_group = [curr_node]
        grouped.append(current_group)

        # 4. Assign stages group-by-group
        num_groups = len(grouped)
        chunk_size = math.ceil(num_groups / num_stages)

        idx = 0
        stage_idx = 0
        while idx < num_groups:
            chunk = grouped[idx : idx + chunk_size]
            for group in chunk:
                for node in group:
                    model_dict[node]['stage'] = stage_idx
            idx += chunk_size
            stage_idx += 1
            # If we exceed num_stages, put all remaining groups on the last stage.
            if stage_idx >= num_stages:
                while idx < num_groups:
                    for node in grouped[idx]:
                        model_dict[node]['stage'] = num_stages - 1
                    idx += 1

        return model_dict


    def as_model_dict(self):
        """
        Symbolically traces 'model' and returns a dictionary describing each
        sub-component (node) with fields: 'callable', 'dst', 'rcv', 'stage',
        'num_layer_shards'.
        """
        if self.model_dict is not None:
            return self.model_dict
        
        traced = fx.symbolic_trace(self)
        graph_dict = {}
        node_names = []
        
        for node in traced.graph.nodes:
            if node.op in ('placeholder', 'output'):
                continue  # Skip input and output nodes

            node_names.append(node.name)

            # Determine sources (rcv) and destinations (dst)
            src_list = [arg.name for arg in node.args if isinstance(arg, fx.Node)]
            dst_list = [user.name for user in node.users]

            # Retrieve module/function info
            if node.op == 'call_module':
                submodule = self.get_submodule(node.target)
                obj = submodule.__class__
                if isinstance(submodule, nn.Conv2d):
                    settings = {
                        'in_channels':  submodule.in_channels,
                        'out_channels': submodule.out_channels,
                        'kernel_size':  submodule.kernel_size,
                        'stride':       submodule.stride,
                        'padding':      submodule.padding,
                        'bias':         submodule.bias is not None
                    }
                elif isinstance(submodule, nn.Linear):
                    settings = {
                        'in_features':  submodule.in_features,
                        'out_features': submodule.out_features,
                        'bias':         submodule.bias is not None
                    }
                else:
                    settings = {k: v for k, v in vars(submodule).items() if not k.startswith('_')}
            elif node.op == 'call_function':
                obj = node.target
                settings = dict(node.kwargs)
            elif node.op == 'call_method':
                obj = f"method_{node.target}"
                settings = dict(node.kwargs)
            else:
                obj = None
                settings = {}

            graph_dict[node.name] = {
                'callable': {
                    'object': obj,
                    'settings': settings
                },
                'dst': {
                    'to': dst_list
                },
                'rcv': {
                    'src': src_list,
                    'strategy': None
                },
                'stage': 0,
                'num_layer_shards': 1
            }
        
        # Identify first and last node keys
        first_node = node_names[0]
        last_node = node_names[-1]

        # Rename their entries in graph_dict
        graph_dict["start"] = graph_dict.pop(first_node)
        graph_dict["finish"] = graph_dict.pop(last_node)

        # Update references in each node’s 'rcv' and 'dst'
        for k, v in graph_dict.items():
            v["dst"]["to"] = [
                "start" if x == first_node else "finish" if x == last_node else x
                for x in v["dst"]["to"]
            ]
            v["rcv"]["src"] = [
                "start" if x == first_node else "finish" if x == last_node else x
                for x in v["rcv"]["src"]
            ]

        # Clear 'src' in "start" and 'to' in "finish"
        graph_dict["start"]["rcv"]["src"] = []
        graph_dict["finish"]["dst"]["to"] = []

        for info in graph_dict.values():
            if "callable" in info and "settings" in info["callable"]:
                info["callable"]["settings"].pop("training", None)

        self.model_dict = BaseModel.set_stage(graph_dict, self.config.num_stages)
        # TODO: when you have multiple stages, make sure to bundle callables with object type 'function' 
        # in them together in a single stage and not separate them if possible.
        return self.set_stage(graph_dict, self.config.num_stages)
        
        