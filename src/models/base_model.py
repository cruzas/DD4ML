import math
from abc import ABC, abstractmethod
from collections import deque

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
    
    def set_stage(self, model_dict, num_stages):
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

    