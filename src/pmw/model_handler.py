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


def preprocessing(batch):
    # flatten the batch
    batch = nn.flatten(batch)
    return batch[:, :700], batch[:, 700:]


def average_fun(input1, input2):
    return (input1+input2)/2


class ModelHandler():
    def __init__(self, net_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None):
        # TODO: Add a security check to ensure that the network has valid "to", "src", and stage numbers
        self.net_dict = net_dict
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

    def __str__(self):
        result = []
        for stage, layers in self.organized_layers.items():
            result.append(f"Stage {stage}:")
            for layer in layers:
                result.append(f"\t{layer}")
        return "\n".join(result)

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
        stages = {}
        # Group layers by stage
        for layer_name, layer_info in net.items():
            stage = layer_info.get('stage')
            if stage is None:
                continue
            stages.setdefault(stage, []).append(layer_name)

        organized_layers = {}
        num_ranks_per_model = 0
        for stage, layers in stages.items():
            # Build dependency graph for layers in this stage
            graph = {layer: []
                     for layer in layers}  # Initialize adjacency list
            num_layer_shards = None
            layer_name = ''
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
                old_num_layer_shards = layer_info['num_layer_shards'] if num_layer_shards is None else num_layer_shards
                num_layer_shards = layer_info['num_layer_shards']
                old_layer_name = layer_name
                layer_name = layer
                if num_layer_shards != old_num_layer_shards:
                    raise ValueError(f"Layer '{layer}' has a different number of layer shards ({num_layer_shards}) than the previous layer '{old_layer_name}' ({old_num_layer_shards}). Note that every layer in a stage must have the same number of layer shards.")
                    
            num_ranks_per_model += num_layer_shards

            # Perform topological sort on the graph
            try:
                ordered_layers = self._topological_sort(graph)
            except Exception as e:
                print(f"Error in stage {stage}: {e}")
                ordered_layers = []
            organized_layers[stage] = ordered_layers

        return organized_layers, num_ranks_per_model

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
