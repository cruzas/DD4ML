import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import copy

def preprocessing(batch):
    # flatten the batch
    batch = nn.flatten(batch)
    return batch[:,:700], batch[:,700:]

def average_fun(input1, input2):
    return (input1+input2)/2

net = {
    "start": {
        "callable": {'object': preprocessing, 'settings': {}},
        "dst": {"to": ["input_layer1_1", "input_layer1_2"], "strategy": None},
        "rcv": {"src": [], "strategy": None},
        "stage": 1,
    },
    "input_layer1_1": {
        "callable": {'object': nn.Linear, 'settings': {"in_features": 700, "out_features": 256}},
        "dst": {'to': ["input_layer2_1"], 'strategy': None}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
        "rcv": {'src': ["start"], 'strategy': None},
        "stage": 1, 
    },
    "input_layer1_2": {
        "callable": {'object': nn.Linear, "settings": {"in_features": 84, "out_features": 32}},
        "dst": {'to': ["input_layer2_2"], 'strategy': None},
        "rcv": {"src": ["start"], "strategy": None},
        "stage": 2, 
    },  
    "input_layer2_1": {
        "callable": {'object': nn.Linear, 'settings': {"in_features": 256, "out_features": 128}},
        "dst": {'to': ["finish"], 'strategy': None}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
        "rcv": {'src': ["input_layer1_1"], 'strategy': None},
        "stage": 1, 
    },
    "input_layer2_2": {
        "callable": {'object': nn.Linear, "settings": {"in_features": 32, "out_features": 128}},
        "dst": {'to': ["finish"], 'strategy': None},
        "rcv": {"src": ["input_layer1_2"], "strategy": None},
        "stage": 2, 
    },
    "finish": {
        "callable": {'object': nn.Linear, "settings": {"in_features": 128, "out_features": 10}},
        "dst": {"to": [], "strategy": None},
        "rcv": {'src': ["input_layer2_1", "input_layer2_2"], 'strategy': average_fun},
        "stage": 2,
    },
}

class NetHandler():
    def __init__(self, net_dict):
        # TODO: Add a security check to ensure that the network has valid "to", "src", and stage numbers
        self.net_dict = net_dict
        self._validate_network() 
        self.organized_layers = self._organize_layers()
        self.stage_list = self._get_stage_list()
    
    def __str__(self):
        result = []
        for stage, layers in self.organized_layers.items():
            result.append(f"Stage {stage}:")
            for layer in layers:
                result.append(f"\t{layer}")
        return "\n".join(result)

    def _get_stage_list(self):
        # Return a list of lists, where each sublist contains the layers in a stage
        stage_list = list(self.organized_layers.values())
        if "start" not in stage_list[0]:
            # Move the stage with the "start" layer to the beginning
            for i, stage in enumerate(stage_list):
                if "start" in stage:
                    stage_list.insert(0, stage_list.pop(i))
                    break
        if "finish" not in stage_list[-1]:
            for i, stage in enumerate(stage_list):
                if "finish" in stage:
                    stage_list.append(stage_list.pop(i))
                    break
        return stage_list

    def _organize_layers(self):
        net = self.net_dict
        stages = {}
        # Group layers by stage
        for layer_name, layer_info in net.items():
            stage = layer_info.get('stage')
            if stage is None:
                continue
            stages.setdefault(stage, []).append(layer_name)

        organized_layers = {}
        for stage, layers in stages.items():
            # Build dependency graph for layers in this stage
            graph = {layer: [] for layer in layers}  # Initialize adjacency list

            for layer in layers:
                layer_info = net[layer]
                rcv_sources = layer_info.get('rcv', {}).get('src', [])
                if rcv_sources is None:
                    rcv_sources = []
                for src in rcv_sources:
                    if src in layers:
                        # Dependency within the same stage
                        # Edge from src to layer (src -> layer)
                        graph[src].append(layer)

            # Perform topological sort on the graph
            try:
                ordered_layers = self._topological_sort(graph)
            except Exception as e:
                print(f"Error in stage {stage}: {e}")
                ordered_layers = []
            organized_layers[stage] = ordered_layers
        return organized_layers

    def _topological_sort(self, graph):
        visited = set()
        temp_marks = set()
        result = []

        def visit(node):
            if node in temp_marks:
                raise Exception("Graph has cycles")
            if node not in visited:
                temp_marks.add(node)
                for m in graph.get(node, []):
                    visit(m)
                temp_marks.remove(node)
                visited.add(node)
                result.insert(0, node)  # Prepend to result to get correct order

        for node in graph:
            if node not in visited:
                visit(node)

        return result

    def _validate_network(self):
        
        net = copy.deepcopy(self.net_dict)
        for k,v in net.items():
            v["stage"] = 1

        # check that the layers names do not contain symbols which cannot be in a python variable name
        for layer_name in net.keys():
            if not layer_name.isidentifier():
                raise ValueError(f"Layer '{layer_name}' contains invalid characters. Only alphanumeric characters and underscores are allowed.")
        errors = []

        # Check for missing 'rcv' or 'dst', and invalid references
        for layer_name, layer_info in net.items():
            # Check for 'rcv' and 'dst' keys
            if 'rcv' not in layer_info:
                errors.append(f"Layer '{layer_name}' is missing 'rcv' entry.")
                continue  # Skip further checks for this layer
            if 'dst' not in layer_info:
                errors.append(f"Layer '{layer_name}' is missing 'dst' entry.")
                continue

            # Check that 'rcv' sources exist
            rcv_sources = layer_info['rcv'].get('src', [])
            if rcv_sources is None:
                rcv_sources = []
            for src in rcv_sources:
                if src not in net:
                    errors.append(f"Layer '{layer_name}' has 'rcv' source '{src}' which does not exist.")

            # Check that 'dst' destinations exist
            dst_targets = layer_info['dst'].get('to', [])
            if dst_targets is None:
                dst_targets = []
            for dst in dst_targets:
                if dst not in net:
                    errors.append(f"Layer '{layer_name}' has 'dst' target '{dst}' which does not exist.")

            # Check for mutual consistency between 'rcv' and 'dst'
            for dst in dst_targets:
                dst_rcv_sources = net[dst]['rcv'].get('src', [])
                if layer_name not in dst_rcv_sources:
                    errors.append(f"Layer '{layer_name}' lists '{dst}' as a destination, but '{dst}' does not have '{layer_name}' in its 'rcv' sources.")

            for src in rcv_sources:
                src_dst_targets = net[src]['dst'].get('to', [])
                if layer_name not in src_dst_targets:
                    errors.append(f"Layer '{layer_name}' lists '{src}' as a source, but '{src}' does not have '{layer_name}' in its 'dst' targets.")

        # Check for cycles in dependency graphs within stages
        stages = {}
        # Group layers by stage
        for layer_name, layer_info in net.items():
            stage = layer_info.get('stage')
            if stage is None:
                errors.append(f"Layer '{layer_name}' is missing 'stage' entry.")
                continue
            stages.setdefault(stage, []).append(layer_name)

        for stage, layers in stages.items():
            # Build dependency graph for layers in this stage
            graph = {layer: [] for layer in layers}
            for layer in layers:
                layer_info = net[layer]
                rcv_sources = layer_info['rcv'].get('src', [])
                if rcv_sources is None:
                    rcv_sources = []
                for src in rcv_sources:
                    if src in layers:
                        graph[src].append(layer)

            # Check for cycles
            try:
                self._topological_sort(graph)
            except Exception as e:
                errors.append(f"Cycle detected in stage {stage}: {e}")

        if errors:
            temp = '\n'.join(errors)
            raise ValueError(f"Network validation failed. See list of errors:\n{temp}")

nh = NetHandler(net)
# print(nh)
print(nh.stage_list)

