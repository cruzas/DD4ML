import math
from abc import ABC, abstractmethod
from collections import deque

import torch.fx as fx
import torch.nn as nn

from src.utils import CfgNode as CN


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
        In case of a model defined via a model dictionary.
        
        Assign pipeline stages to each module in model_dict based on a 
        topological sort from 'start' to 'finish'. Distribute them as 
        evenly as possible among the stages.
        """
        # 1. Build adjacency (forward edges) and in-degree count
        adjacency = {}  # adjacency[u] = list of nodes that receive from u
        in_degree = {}  # how many edges point to each node

        for key, info in model_dict.items():
            adjacency[key] = info['dst']['to']  # children in the forward pass
            in_degree[key] = 0                 # initialize in-degree

        # Calculate in-degree of each node
        for src, children in adjacency.items():
            for child in children:
                in_degree[child] += 1

        # 2. Topological sort using Kahn's Algorithm
        #    Initialize queue with nodes that have in-degree == 0
        queue = deque([k for k in in_degree if in_degree[k] == 0])
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            # Decrement in-degree of all children
            for child in adjacency[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # (Optional) Check that we actually saw all modules.
        # If you expect to have visited all keys, but topo_order is
        # missing some, it might mean there's a cycle somewhere.
        if len(topo_order) < len(model_dict):
            raise ValueError("Cycle detected or unreachable nodes exist in the graph!")

        # 3. Distribute the modules across stages as evenly as possible.
        num_modules = len(topo_order)
        chunk_size = math.ceil(num_modules / num_stages)

        # We'll iterate in slices of 'chunk_size' and assign them to successive stages.
        idx = 0
        stage_idx = 0
        while idx < num_modules:
            # Get the slice for the current stage
            chunk = topo_order[idx : idx + chunk_size]
            # Assign them to the current stage
            for key in chunk:
                model_dict[key]['stage'] = stage_idx

            idx += chunk_size
            stage_idx += 1

            if stage_idx >= num_stages:
                # If we run out of stage slots, just put remaining modules on the last stage
                while idx < num_modules:
                    model_dict[topo_order[idx]]['stage'] = num_stages - 1
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

        self.model_dict = BaseModel.set_stage(graph_dict, self.config.num_stages)
        # TODO: when you have multiple stages, make sure to bundle callables with object type 'function' 
        # in them together in a single stage and not separate them if possible.
        return self.set_stage(graph_dict, self.config.num_stages)
        
        