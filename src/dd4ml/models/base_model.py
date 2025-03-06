import math
from abc import ABC
from collections import deque

import torch.nn as nn

from dd4ml.utility import CfgNode as CN
from dd4ml.utility import is_function_module


class BaseModel(nn.Module, ABC):
    n_layer = None

    @staticmethod
    def get_default_config():
        C = CN()
        # In case of using pmw
        C.num_stages = None
        C.num_subdomains = 1
        C.num_replicas_per_subdomain = 1
        return C

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.model_dict = None

    def set_stage(self):
        """
        Topologically sort modules, then assign them to pipeline stages.
        Consecutive 'function-like' modules (e.g. <function relu at 0x...>)
        are grouped into a single bundle so they share the same stage.
        """

        # 1. Build adjacency (forward edges) & in-degree
        adjacency = {}
        in_degree = {}
        for key, info in self.model_dict.items():
            adjacency[key] = info["dst"]["to"]
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

        if len(topo_order) < len(self.model_dict):
            raise ValueError("Cycle or unreachable nodes in the graph!")

        # 3. Merge consecutive function modules
        grouped = []
        current_group = [topo_order[0]]

        for i in range(1, len(topo_order)):
            prev_node = topo_order[i - 1]
            curr_node = topo_order[i]
            # Check if both are function modules and curr is the sole child of prev.
            if (
                is_function_module(self.model_dict[prev_node])
                and is_function_module(self.model_dict[curr_node])
                and adjacency[prev_node] == [curr_node]
            ):
                current_group.append(curr_node)
            else:
                grouped.append(current_group)
                current_group = [curr_node]
        grouped.append(current_group)

        # 4. Assign stages group-by-group
        num_groups = len(grouped)
        chunk_size = math.ceil(num_groups / self.config.num_stages)

        idx = 0
        stage_idx = 0
        while idx < num_groups:
            chunk = grouped[idx : idx + chunk_size]
            for group in chunk:
                for node in group:
                    self.model_dict[node]["stage"] = stage_idx
            idx += chunk_size
            stage_idx += 1
            # If we exceed self.config.num_stages, put all remaining groups on the last stage.
            if stage_idx >= self.config.num_stages:
                while idx < num_groups:
                    for node in grouped[idx]:
                        self.model_dict[node]["stage"] = self.config.num_stages - 1
                    idx += 1

        return self.model_dict

    # @abstractmethod
    # def as_model_dict(self):
    #     # Was a bit too difficult to implement. Maybe TODO later
    #     pass
