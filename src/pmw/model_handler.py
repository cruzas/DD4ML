import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import json
import os
import sys

def preprocessing(batch):
    # flatten the batch
    batch = nn.flatten(batch)
    return batch[:,:700], batch[:,700:]

def average_fun(input1, input2):
    return (input1+input2)/2

net = {
    "start": {
        "callable": {'object': preprocessing, 'settings': {}},
        "dst": {"to": ["input_layer1.1", "input_layer1.2"], "strategy": None},
        "rcv": {"src": None, "strategy": None},
        "stage": 1,
    },
    "input_layer1.1": {
        "callable": {'object': nn.Linear, 'settings': {"in_features": 700, "out_features": 256}},
        "dst": {'to': ["input_layer2.1"], 'strategy': None}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
        "rcv": {'src': ["start"], 'strategy': None},
        "stage": 1, 
    },
    "input_layer1.2": {
        "callable": {'object': nn.Linear, "settings": {"in_features": 84, "out_features": 32}},
        "dst": {'to': ["input_layer2.2"], 'strategy': None},
        "rcv": {"src": ["start"], "strategy": None},
        "stage": 2, 
    },  
    "input_layer2.1": {
        "callable": {'object': nn.Linear, 'settings': {"in_features": 256, "out_features": 128}},
        "dst": {'to': ["finish"], 'strategy': None}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
        "rcv": {'src': ["input_layer1.1"], 'strategy': None},
        "stage": 1, 
    },
    "input_layer2.2": {
        "callable": {'object': nn.Linear, "settings": {"in_features": 32, "out_features": 128}},
        "dst": {'to': ["finish"], 'strategy': None},
        "rcv": {"src": ["input_layer1.2"], "strategy": None},
        "stage": 2, 
    },
    "finish": {
        "callable": {'object': nn.Linear, "settings": {"in_features": 128, "out_features": 10}},
        "dst": {"to": None, "strategy": None},
        "rcv": {'src': ["input_layer2.1"], 'strategy': average_fun},
        "stage": 2,
    },
}

class NetHandler():
    def __init__(self, net_dict):
        # TODO: Add a security check to ensure that the network has valid "to", "src", and stage numbers
        self.net_dict = net_dict
    
    # def find_first_layers(self):
    #     net = self.net_dict
    #     stages = {}
    #     # Group layers by stage
    #     for layer_name, layer_info in net.items():
    #         stage = layer_info.get('stage')
    #         if stage is None:
    #             continue
    #         stages.setdefault(stage, []).append(layer_name)

    #     first_layers = {}
    #     for stage, layers in stages.items():
    #         first_layers_in_stage = []
    #         for layer_name in layers:
    #             layer_info = net[layer_name]
    #             rcv_sources = layer_info.get('rcv', {}).get('src', [])
    #             if rcv_sources is None or not rcv_sources:
    #                 # No sources; it's a first layer
    #                 first_layers_in_stage.append(layer_name)
    #             else:
    #                 # Check if any source is in the same stage
    #                 src_stages = [net[src].get('stage') for src in rcv_sources if src in net]
    #                 if all(src_stage != stage for src_stage in src_stages):
    #                     first_layers_in_stage.append(layer_name)
    #         first_layers[stage] = first_layers_in_stage
    #     return first_layers                

    def organize_layers(self):
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
                ordered_layers = self.topological_sort(graph)
            except Exception as e:
                print(f"Error in stage {stage}: {e}")
                ordered_layers = []
            organized_layers[stage] = ordered_layers

        return organized_layers

    def topological_sort(self, graph):
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



nh = NetHandler(net)
first_layers = nh.organize_layers()

# Print the layers in nh.net_dict
for stage, layers in first_layers.items():
    print(f"Stage {stage}:")
    for layer in layers:
        print(f"  {layer}")