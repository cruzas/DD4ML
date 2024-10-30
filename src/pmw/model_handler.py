import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import copy
import utils
from collections import deque

def max_path_length(model, layer):
    '''
    Maximum distance between "layer" and "finish" in the model
    '''
    if layer == 'finish':
        return 0
    return 1 + max([max_path_length(model, dst) for dst in model[layer]['dst']['to']])

def resolve_backward_dependencies(model):
    start = 'finish'
    max_distances = {layer: max_path_length(model, layer) for layer in model.keys()}
    max_distances = list(dict(sorted(max_distances.items(), key=lambda item: item[1], reverse=False)).keys())
    done = []
    while True:
        dst_layers = model[start]['rcv']['src']
        max_path_lengths = {layer: max_path_length(model, layer) for layer in dst_layers}
        max_path_lengths_decreasing = dict(sorted(max_path_lengths.items(), key=lambda item: item[1], reverse=True))
        max_path_lengths_decreasing = list(max_path_lengths_decreasing.keys())
        model[start]['bwd_dst'] = {'to': max_path_lengths_decreasing}
        for layer in max_path_lengths_decreasing:
            model[layer].setdefault('bwd_rcv', {'src': []})
            model[layer]['bwd_rcv']['src'].append(start)
        done.append(start)
        max_distances.pop(0)
        if not max_distances:
            break
        start = max_distances[0]
    return model

def resolve_layer_dependencies(model):
    # Build a dependency graph and count incoming edges (src dependencies) for each node
    dependencies = {layer: set(info['rcv']['src']) for layer, info in model.items()}
    dependents = {layer: set(info['dst']['to']) for layer, info in model.items()}
    incoming_edges_count = {layer: len(srcs) for layer, srcs in dependencies.items()}

    # Start with nodes with no incoming edges (no dependencies)
    to_process = deque([layer for layer, count in incoming_edges_count.items() if count == 0])
    ordered_layers = []
    
    # Perform topological sort
    while to_process:
        layer = to_process.popleft()
        ordered_layers.append(layer)
        
        # Reduce incoming edges for each dependent layer
        for dependent in dependents[layer]:
            incoming_edges_count[dependent] -= 1
            if incoming_edges_count[dependent] == 0:
                to_process.append(dependent)
    
    if len(ordered_layers) != len(model):
        raise ValueError("Dependency cycle detected in the model layers.")
    
    # Reorder `dst['to']` and `rcv['src']` based on the topological sort
    layer_position = {layer: pos for pos, layer in enumerate(ordered_layers)}
    for layer, info in model.items():
        # Sort `dst['to']` based on order in the topological sort
        info['dst']['to'].sort(key=lambda x: layer_position[x])
        # Sort `rcv['src']` based on order in the topological sort
        info['rcv']['src'].sort(key=lambda x: layer_position[x])

    return model

def preprocessing(batch):
    # flatten the batch
    batch = nn.flatten(batch)
    return batch[:, :700], batch[:, 700:]


def average_fun(input1, input2):
    return (input1+input2)/2


class ModelHandler():
    def __init__(self, net_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None):
        # TODO: Add a security check to ensure that the network has valid "to", "src", and stage numbers
        self.net_dict = resolve_layer_dependencies(net_dict)
        self.num_subdomains = num_subdomains
        self.num_replicas_per_subdomain = num_replicas_per_subdomain
        self.tot_replicas = num_subdomains*num_replicas_per_subdomain
        self.available_ranks = sorted(available_ranks) if available_ranks is not None else list(range(dist.get_world_size()))
        self.global_model_group = dist.new_group(self.available_ranks, use_local_synchronization=True)
        self.rank = dist.get_rank()
        
        self._validate_network()
        self.organized_layers, self.num_ranks_per_model = self._organize_layers()
        self.stage_list = self._get_stage_list()
        self.num_stages = len(self.stage_list)
        self.nn_structure = self.create_distributed_model_rank_structure()
        self.rank_to_position() # Initializes self.sd, self.rep, self.s, self.sh
        self._stage_data = self.stage_data()
        self.get_list_of_consecutive_layers()
        self.net_dict = resolve_backward_dependencies(net_dict)
    

    def __str__(self):
        result = []
        for stage, layers in self.organized_layers.items():
            result.append(f"Stage {stage}:")
            for layer in layers:
                result.append(f"\t{layer}")
        return "\n".join(result)
    
    def get_list_of_consecutive_layers(self):
        '''
        Checking for non-consecutive layers in a stage. This is not allowed as it would increase the complexity of the implementation and CPU time.
        '''
        lst = [[]]
        consecutive_layer = True
        for layer_name in self._stage_data['layers']:
            if consecutive_layer:
                lst[-1].append(layer_name)
            else:
                lst.append([layer_name])
            consecutive_layer = False
            for dst_name in self.net_dict[layer_name]['dst']['to']:
                current_layer_stage = self.net_dict[layer_name]['stage']
                src_layer_stage = self.net_dict[dst_name]['stage']
                if current_layer_stage == src_layer_stage:
                    consecutive_layer = True
        if len(lst) != 1:
            raise ValueError("The layers in a stage are not consecutive.")
        return lst
                    
    def get_stage_ranks(self, stage_name, mode):
        assert mode in ['local', 'global', 'replica'], f"Invalid mode '{mode}'. Must be either 'global', 'local', or 'replica'."
        assert stage_name in ['first', 'last'], f"Invalid stage '{stage_name}'. Must be either 'first' or 'last'."
        stage = 0 if stage_name == 'first' else self.num_stages - 1
        sd, rep, _, _ = self.rank_to_position()
        stage_ranks = []

        def _go_through_replicas(sd, stage, stage_ranks):
            for rep in range(self.num_replicas_per_subdomain):
                ranks = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{stage}"]["ranks"]
                assert len(ranks) == 1, f"Tensor sharding not implemented yet. Expected 1 rank, got {len(ranks)}."
                stage_ranks.append(ranks[0])
            return stage_ranks

        if not hasattr(self, f'{stage_name}_stage_ranks_{mode}'):
            if mode == 'replica':
                stage_ranks = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{stage}"]["ranks"]
            elif mode == 'local':
                stage_ranks = _go_through_replicas(sd, stage, stage_ranks)
            elif mode == 'global':
                for sd in range(self.num_subdomains):
                    stage_ranks = _go_through_replicas(sd, stage, stage_ranks)
            setattr(self, f'{stage_name}_stage_ranks_{mode}', stage_ranks)
        return getattr(self, f'{stage_name}_stage_ranks_{mode}')
    
    def is_first_stage(self):
        sd, rep, _, _ = self.rank_to_position()
        return self.rank in self.nn_structure[f"sd{sd}"][f"r{rep}"]["s0"]["ranks"]

    def is_last_stage(self):
        sd, rep, _, _ = self.rank_to_position()
        s_final = self.num_stages - 1
        return self.rank in self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s_final}"]["ranks"]

    def subdomain_ranks(self):
        for sd in range(self.num_subdomains):
            if self.rank in self.nn_structure[f"sd{sd}"]["ranks"]:
                return self.nn_structure[f"sd{sd}"]["ranks"]
            
    def replica_ranks(self):
        sd, rep, _, _ = self.rank_to_position()
        return self.nn_structure[f"sd{sd}"][f"r{rep}"]["ranks"]
    
    def get_sd_group(self):
        sd, _, _, _ = self.rank_to_position()
        return self.nn_structure[f"sd{sd}"]["group"]

    def get_replica_group(self):
        sd, rep, _, _ = self.rank_to_position()
        return self.nn_structure[f"sd{sd}"][f"r{rep}"]["group"]
        
    def get_layers_copy_group(self, mode='global'):
        if mode not in ['local', 'global']:
            raise ValueError(f"Invalid mode '{mode}'. Must be either 'global' or 'local'.")
        sd, rep, s, sh = self.rank_to_position()
        return self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"][f"sh{sh}"][f"{mode}_group"]
            
    def stage_data(self):
        # Get the stage data corresponding to this rank
        for sd in range(self.num_subdomains):
            for rep in range(self.num_replicas_per_subdomain):
                for s in range(len(self.stage_list)):
                    if self.rank in self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["ranks"]:
                        return self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]
        
    def rank_to_position(self):
        # Get the subdomain and replica number correspoding to this rank
        if hasattr(self, 'sd'):
            return self.sd, self.rep, self.s, self.sh
        else:
            for sd in range(self.num_subdomains):
                for rep in range(self.num_replicas_per_subdomain):
                    for s in range(self.num_stages):
                        layer_names = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["layers"]
                        num_shards = self.net_dict[layer_names[0]]["num_layer_shards"]
                        for sh in range(num_shards):
                            if self.rank == self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"][f"sh{sh}"]["rank"]:
                                self.sd, self.rep, self.s, self.sh = sd, rep, s, sh
                                return sd, rep, s, sh
    
    def layer_name_to_ranks(self, layer_name): 
        # within the same subdomain and replica
        sd, rep, _, _ = self.rank_to_position()
        for s in range(self.num_stages):
            if layer_name in self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["layers"]:
                return self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["ranks"]

    def create_distributed_model_rank_structure(self):
        if len(self.available_ranks) < self.num_subdomains*self.num_replicas_per_subdomain*self.num_ranks_per_model:
            raise ValueError(
                f"Number of available ranks ({len(self.available_ranks)}) is less than the required number of ranks ({self.num_subdomains*self.num_replicas_per_subdomain*self.num_ranks_per_model}).")
        elif len(self.available_ranks) > self.num_subdomains*self.num_replicas_per_subdomain*self.num_ranks_per_model:
            print(
                f"Warning: Number of available ranks ({len(self.available_ranks)}) is more than the required number of ranks ({self.num_subdomains*self.num_replicas_per_subdomain})... some will be idle.")
            self.available_ranks = self.available_ranks[:self.num_subdomains *
                                              self.num_replicas_per_subdomain*self.num_ranks_per_model]

        # Split self.available_ranks into num_subdomains chunks
        n = self.num_replicas_per_subdomain*self.num_ranks_per_model
        subdomain_ranks = [self.available_ranks[i*n:(i+1)*n] for i in range(0, self.num_subdomains)]
        # TODO: Check whether the ranks are ordered correctly with respect to the node and gpus (e.g. ranks 0,1,2,3 are the ranks on the same node with 4 gpus, ...)
        # in this case use function utils.gather_node_info()
        nn_structure = {}
        nc = self.num_subdomains*self.num_replicas_per_subdomain # network copies
        for sd in range(self.num_subdomains):
            nn_structure[f"sd{sd}"] = {"ranks": subdomain_ranks[sd],
                                       "group": dist.new_group(subdomain_ranks[sd], use_local_synchronization=True)}
            for rep in range(self.num_replicas_per_subdomain):
                # split the ranks into num_replicas_per_subdomain chunks
                rep_ranks = subdomain_ranks[sd][rep*self.num_ranks_per_model:(rep+1)*self.num_ranks_per_model]
                nn_structure[f"sd{sd}"][f"r{rep}"] = {"ranks": rep_ranks,
                                                      "group": dist.new_group(rep_ranks, use_local_synchronization=True)}
                model_ranks = nn_structure[f"sd{sd}"][f"r{rep}"]["ranks"]
                old_ranks_per_this_stage = 0
                for s, (stage, layers_in_stage) in enumerate(self.organized_layers.items()):
                    ranks_per_this_stage = self.net_dict[layers_in_stage[0]]["num_layer_shards"]
                    nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"] = {"ranks": model_ranks[old_ranks_per_this_stage:old_ranks_per_this_stage+ranks_per_this_stage], "layers": layers_in_stage}
                
                    old_ranks_per_this_stage += ranks_per_this_stage
                    
                    for sh, rank in zip(range(self.net_dict[layers_in_stage[0]]["num_layer_shards"]), nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["ranks"]):  
                        ranks_on_this_stage = nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["ranks"]  
                        if sd == 0 and rep == 0:
                            global_ranks = [rank + i*len(model_ranks) for i in range(nc)] # Ranks to synchronize stage with on all subdomains
                            global_group = dist.new_group(global_ranks, use_local_synchronization=True)
                        else:
                            global_ranks = nn_structure[f"sd0"][f"r0"][f"s{s}"][f"sh{sh}"]["global_ranks"]
                            global_group = nn_structure[f"sd0"][f"r0"][f"s{s}"][f"sh{sh}"]["global_group"]
                        if rep == 0:
                            local_ranks = [rank + i*self.num_stages for i in range(self.num_replicas_per_subdomain)] # Ranks to synchronize stage with on this subdomain
                            local_group = dist.new_group(local_ranks, use_local_synchronization=True)
                        else:
                            local_ranks = nn_structure[f"sd{sd}"][f"r0"][f"s{s}"][f"sh{sh}"]["local_ranks"]
                            local_group = nn_structure[f"sd{sd}"][f"r0"][f"s{s}"][f"sh{sh}"]["local_group"]

                        nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"][f"sh{sh}"] = { 
                            "global_ranks": global_ranks,
                            "local_ranks": local_ranks,
                            "shard_ranks": ranks_on_this_stage, # Ranks to shard the stage with
                            "rank": rank,
                            "global_group": global_group, # Group for global synchronization
                            "local_group": local_group, # Group for local synchronization
                        }
        return nn_structure
    
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
        '''
        Organize layers into stages and build dependency graph within each stage
        '''
        net = self.net_dict
        organized_layers = {}
        
        dst = net['start']['dst']['to']
        organized_layers[net['start']['stage']] = ['start']
        while dst:
            next_dst = []
            for layer_name in dst:
                organized_layers.setdefault(net[layer_name]['stage'], [])
                if layer_name not in organized_layers[net[layer_name]['stage']]:
                    organized_layers[net[layer_name]['stage']].append(layer_name)
                next_dst.extend(net[layer_name]['dst']['to'])
            dst = next_dst
            
        # make it so the 'start' layer is on key 0 and the 'finish' layer is on the last key
        dict2 = {}
        counter = 0
        for key, value in organized_layers.items():
            if 'start' in value:
                dict2[0] = value
            if 'finish' not in value:
                dict2[counter] = value
            else:
                finish_key = key
                counter -= 1
            counter += 1
        dict2[counter] = organized_layers[finish_key]
        return dict2, len(dict2.keys())

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
                # Prepend to result to get correct order
                result.insert(0, node)

        for node in graph:
            if node not in visited:
                visit(node)

        return result

    def _validate_network(self):

        net = copy.deepcopy(self.net_dict)
        
        #first make sure that there are layers called 'start' and 'finish'
        if 'start' not in net:
            raise ValueError("Network must have a layer called 'start'.")
        if 'finish' not in net:
            raise ValueError("Network must have a layer called 'finish'.")
        
        for k, v in net.items():
            v["stage"] = 1 # This is done to check for cycles in the network graph.

        # check that the layers names do not contain symbols which cannot be in a python variable name
        for layer_name in net.keys():
            if not layer_name.isidentifier():
                raise ValueError(
                    f"Layer '{layer_name}' contains invalid characters. Only alphanumeric characters and underscores are allowed.")
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
            rcv_strategy = layer_info['rcv'].get('strategy')
            if rcv_sources is None:
                rcv_sources = []
            else:
                if len(rcv_sources) > 1 and rcv_strategy is None:
                    errors.append(f"Layer '{layer_name}' has multiple receive sources but no strategy.")

            for src in rcv_sources:
                if src not in net:
                    errors.append(
                        f"Layer '{layer_name}' has 'rcv' source '{src}' which does not exist.")

            # Check that 'dst' destinations exist
            dst_targets = layer_info['dst'].get('to', [])
            if dst_targets is None:
                dst_targets = []
            for dst in dst_targets:
                if dst not in net:
                    errors.append(
                        f"Layer '{layer_name}' has 'dst' target '{dst}' which does not exist.")

            # Check for mutual consistency between 'rcv' and 'dst'
            for dst in dst_targets:
                dst_rcv_sources = net[dst]['rcv'].get('src', [])
                if layer_name not in dst_rcv_sources:
                    errors.append(
                        f"Layer '{layer_name}' lists '{dst}' as a destination, but '{dst}' does not have '{layer_name}' in its 'rcv' sources.")

            for src in rcv_sources:
                src_dst_targets = net[src]['dst'].get('to', [])
                if layer_name not in src_dst_targets:
                    errors.append(
                        f"Layer '{layer_name}' lists '{src}' as a source, but '{src}' does not have '{layer_name}' in its 'dst' targets.")

        # Check for cycles in dependency graphs within stages
        stages = {}
        # Group layers by stage
        for layer_name, layer_info in net.items():
            stage = layer_info.get('stage')
            if stage is None:
                errors.append(
                    f"Layer '{layer_name}' is missing 'stage' entry.")
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
            raise ValueError(
                f"Network validation failed. See list of errors:\n{temp}")

        
# nh = NetHandler(net)
# print(nh)
# print(nh.stage_list)
