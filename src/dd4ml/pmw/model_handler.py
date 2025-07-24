import copy
import pprint

import torch.distributed as dist
import torch.nn as nn

from dd4ml.utility import broadcast_dict, dprint


def max_path_length(model, layer, cache=None):
    """
    Compute the maximum distance between `layer` and 'finish', using an explicit cache.
    """
    if cache is None:
        cache = {}
    if layer in cache:
        return cache[layer]
    if layer == "finish":
        cache[layer] = 0
        return 0
    value = 1 + max(
        max_path_length(model, dst, cache) for dst in model[layer]["dst"]["to"]
    )
    cache[layer] = value
    return value


def resolve_backward_dependencies(model):
    cache = {}
    start = "finish"
    max_distances = {
        layer: max_path_length(model, layer, cache) for layer in model.keys()
    }
    max_distances = list(
        dict(sorted(max_distances.items(), key=lambda item: item[1])).keys()
    )
    while True:
        dst_layers = model[start]["rcv"]["src"]
        max_path_lengths = {
            layer: max_path_length(model, layer, cache) for layer in dst_layers
        }
        max_path_lengths_decreasing = dict(
            sorted(max_path_lengths.items(), key=lambda item: item[1], reverse=True)
        )
        sorted_layers = list(max_path_lengths_decreasing.keys())
        model[start]["bwd_dst"] = {"to": sorted_layers}
        for layer in sorted_layers:
            model[layer].setdefault("bwd_rcv", {"src": []})
            model[layer]["bwd_rcv"]["src"].append(start)
        max_distances.pop(0)
        if not max_distances:
            break
        start = max_distances[0]
    return model


def resolve_layer_dependencies(model):
    dependencies = {layer: set(info["rcv"]["src"]) for layer, info in model.items()}
    dependents = {layer: set(info["dst"]["to"]) for layer, info in model.items()}
    incoming_edges_count = {layer: len(srcs) for layer, srcs in dependencies.items()}

    to_process = [layer for layer, count in incoming_edges_count.items() if count == 0]
    ordered_layers = []
    while to_process:
        layer = to_process.pop(0)
        ordered_layers.append(layer)
        for dependent in dependents[layer]:
            incoming_edges_count[dependent] -= 1
            if incoming_edges_count[dependent] == 0:
                to_process.append(dependent)

    if len(ordered_layers) != len(model):
        raise ValueError("Dependency cycle detected in the model layers.")

    layer_position = {layer: pos for pos, layer in enumerate(ordered_layers)}
    for layer, info in model.items():
        info["fwd_dst"] = {"to": list(info["dst"]["to"])}
        info["fwd_rcv"] = {"src": list(info["rcv"]["src"])}
        info["fwd_dst"]["to"].sort(key=lambda x: layer_position[x])
        info["fwd_rcv"]["src"].sort(key=lambda x: layer_position[x])
    return model


def preprocessing(batch):
    batch = nn.flatten(batch)
    return batch[:, :700], batch[:, 700:]


def average_fun(input1, input2):
    return (input1 + input2) / 2


class ModelHandler:
    def __init__(
        self, net_dict, num_subdomains, num_replicas_per_subdomain, available_ranks=None
    ):
        self.available_ranks = (
            sorted(available_ranks)
            if available_ranks is not None
            else list(range(dist.get_world_size()))
        )
        self.global_model_group = dist.new_group(
            self.available_ranks, use_local_synchronization=True
        )
        self.rank = dist.get_rank()
        self.num_subdomains = num_subdomains
        self.num_replicas_per_subdomain = num_replicas_per_subdomain
        self.tot_replicas = num_subdomains * num_replicas_per_subdomain

        dicti = {}
        if dist.get_rank() == 0:
            self.net_dict = resolve_layer_dependencies(net_dict)
            self._validate_network()
            self.organized_layers, self.num_ranks_per_model = self._organize_layers()
            self.stage_list = self._get_stage_list()
            self.num_stages = len(self.stage_list)
            dicti = {
                "net_dict": self.net_dict,
                "organized_layers": self.organized_layers,
                "stage_list": self.stage_list,
                "num_stages": self.num_stages,
                "num_ranks_per_model": self.num_ranks_per_model,
            }
        dicti = broadcast_dict(dicti, src=0)
        for key, value in dicti.items():
            setattr(self, key, value)

        self.nn_structure = self.create_distributed_model_rank_structure()
        self._build_rank_position_cache()  # Build cache for rank-to-position lookup.
        # Initialize and store the position fields.
        try:
            self.sd, self.rep, self.s, self.sh = self._rank_position_cache[self.rank]
            print(
                f"Rank {self.rank}/{dist.get_world_size() - 1} is assigned to SD {self.sd}, Rep {self.rep}, S {self.s}, SH {self.sh}."
            )
        except:
            raise ValueError(
                f"Rank {self.rank}/{dist.get_world_size() - 1} is not assigned to any subdomain, replica, stage, or shard."
            )
        self._ = self.rank_to_position()  # Ensures caching within rank_to_position.
        self._stage_data = self.stage_data()
        self.get_list_of_consecutive_layers()  # Caches the consecutive layers.
        self.net_dict = resolve_backward_dependencies(self.net_dict)

    def __str__(self):
        result = []
        for stage, layers in self.organized_layers.items():
            result.append(f"Stage {stage}:")
            for layer in layers:
                result.append(f"\t{layer}")
        return "\n".join(result)

    def get_list_of_consecutive_layers(self):
        if hasattr(self, "_consecutive_layers"):
            return self._consecutive_layers

        lst = [[]]
        consecutive_layer = True
        for layer_name in self._stage_data["layers"]:
            if consecutive_layer:
                lst[-1].append(layer_name)
            else:
                lst.append([layer_name])
            consecutive_layer = False
            for dst_name in self.net_dict[layer_name]["dst"]["to"]:
                current_layer_stage = self.net_dict[layer_name]["stage"]
                src_layer_stage = self.net_dict[dst_name]["stage"]
                if current_layer_stage == src_layer_stage:
                    consecutive_layer = True
        if len(lst) != 1:
            raise ValueError("The layers in a stage are not consecutive.")
        self._consecutive_layers = lst
        return lst

    def get_stage_ranks(self, stage_name, mode):
        assert mode in ["local", "global", "replica"], f"Invalid mode '{mode}'."
        assert stage_name in ["first", "last"], f"Invalid stage '{stage_name}'."
        stage = 0 if stage_name == "first" else self.num_stages - 1
        sd, rep, _, _ = self.rank_to_position()
        stage_ranks = []

        def _go_through_replicas(sd, stage, stage_ranks):
            for rep in range(self.num_replicas_per_subdomain):
                ranks = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{stage}"]["ranks"]
                assert len(ranks) == 1, "Tensor sharding not implemented yet."
                stage_ranks.append(ranks[0])
            return stage_ranks

        attr_name = f"{stage_name}_stage_ranks_{mode}"
        if not hasattr(self, attr_name):
            if mode == "replica":
                stage_ranks = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{stage}"][
                    "ranks"
                ]
            elif mode == "local":
                stage_ranks = _go_through_replicas(sd, stage, stage_ranks)
            elif mode == "global":
                for sd in range(self.num_subdomains):
                    stage_ranks = _go_through_replicas(sd, stage, stage_ranks)
            setattr(self, attr_name, stage_ranks)
        return getattr(self, attr_name)

    def is_first_stage(self):
        sd, rep, _, _ = self.rank_to_position()
        return self.rank in self.nn_structure[f"sd{sd}"][f"r{rep}"]["s0"]["ranks"]

    def is_last_stage(self):
        sd, rep, _, _ = self.rank_to_position()
        s_final = self.num_stages - 1
        return (
            self.rank in self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s_final}"]["ranks"]
        )

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

    def get_layers_copy_group(self, mode="global"):
        if mode not in ["local", "global"]:
            raise ValueError(f"Invalid mode '{mode}'.")
        sd, rep, s, sh = self.rank_to_position()
        return self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"][f"sh{sh}"][
            f"{mode}_group"
        ]

    def stage_data(self):
        for sd in range(self.num_subdomains):
            for rep in range(self.num_replicas_per_subdomain):
                for s in range(len(self.stage_list)):
                    if (
                        self.rank
                        in self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["ranks"]
                    ):
                        return self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]

    def _build_rank_position_cache(self):
        self._rank_position_cache = {}
        for sd in range(self.num_subdomains):
            for rep in range(self.num_replicas_per_subdomain):
                for s in range(self.num_stages):
                    stage_info = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]
                    for key, info in stage_info.items():
                        if key.startswith("sh"):
                            shard_index = int(key[2:])
                            self._rank_position_cache[info["rank"]] = (
                                sd,
                                rep,
                                s,
                                shard_index,
                            )

    def rank_to_position(self):
        # If already assigned, return the stored values.
        if hasattr(self, "sd"):
            return self.sd, self.rep, self.s, self.sh
        pos = self._rank_position_cache[self.rank]
        self.sd, self.rep, self.s, self.sh = pos
        return pos

    def layer_name_to_ranks(self, layer_name):
        if not hasattr(self, "_layer_to_ranks_cache"):
            self._layer_to_ranks_cache = {}
        if layer_name in self._layer_to_ranks_cache:
            return self._layer_to_ranks_cache[layer_name]
        sd, rep, _, _ = self.rank_to_position()
        for s in range(self.num_stages):
            stage_layers = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["layers"]
            if layer_name in stage_layers:
                result = self.nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["ranks"]
                self._layer_to_ranks_cache[layer_name] = result
                return result
        return None

    def create_distributed_model_rank_structure(self):
        total_required = (
            self.num_subdomains
            * self.num_replicas_per_subdomain
            * self.num_ranks_per_model
        )
        if len(self.available_ranks) < total_required:
            raise ValueError(
                f"Number of available ranks ({len(self.available_ranks)}) is less than required ({total_required})."
            )
        elif len(self.available_ranks) > total_required:
            dprint(
                f"Warning: Some available ranks will remain idle. Available ranks: {len(self.available_ranks)}, Required: {total_required}"
            )
            self.available_ranks = self.available_ranks[:total_required]

        n = self.num_replicas_per_subdomain * self.num_ranks_per_model
        subdomain_ranks = [
            self.available_ranks[i * n : (i + 1) * n]
            for i in range(self.num_subdomains)
        ]
        nn_structure = {}
        nc = self.num_subdomains * self.num_replicas_per_subdomain
        for sd in range(self.num_subdomains):
            nn_structure[f"sd{sd}"] = {
                "ranks": subdomain_ranks[sd],
                "group": dist.new_group(
                    subdomain_ranks[sd], use_local_synchronization=True
                ),
            }
            for rep in range(self.num_replicas_per_subdomain):
                rep_ranks = subdomain_ranks[sd][
                    rep
                    * self.num_ranks_per_model : (rep + 1)
                    * self.num_ranks_per_model
                ]
                nn_structure[f"sd{sd}"][f"r{rep}"] = {
                    "ranks": rep_ranks,
                    "group": dist.new_group(rep_ranks, use_local_synchronization=True),
                }
                model_ranks = nn_structure[f"sd{sd}"][f"r{rep}"]["ranks"]
                old_ranks_per_this_stage = 0
                for s, stage in enumerate(sorted(self.organized_layers.keys())):
                    layers_in_stage = self.organized_layers[stage]
                    ranks_per_this_stage = self.net_dict[layers_in_stage[0]][
                        "num_layer_shards"
                    ]
                    nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"] = {
                        "ranks": model_ranks[
                            old_ranks_per_this_stage : old_ranks_per_this_stage
                            + ranks_per_this_stage
                        ],
                        "layers": layers_in_stage,
                    }
                    old_ranks_per_this_stage += ranks_per_this_stage

                    for sh, rank in zip(
                        range(self.net_dict[layers_in_stage[0]]["num_layer_shards"]),
                        nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"]["ranks"],
                    ):
                        ranks_on_this_stage = nn_structure[f"sd{sd}"][f"r{rep}"][
                            f"s{s}"
                        ]["ranks"]
                        if sd == 0 and rep == 0:
                            global_ranks = [
                                rank + i * len(model_ranks) for i in range(nc)
                            ]
                            global_group = dist.new_group(
                                global_ranks, use_local_synchronization=True
                            )
                        else:
                            global_ranks = nn_structure[f"sd0"]["r0"][f"s{s}"][
                                f"sh{sh}"
                            ]["global_ranks"]
                            global_group = nn_structure[f"sd0"]["r0"][f"s{s}"][
                                f"sh{sh}"
                            ]["global_group"]
                        if rep == 0:
                            local_ranks = [
                                rank + i * self.num_stages
                                for i in range(self.num_replicas_per_subdomain)
                            ]
                            local_group = dist.new_group(
                                local_ranks, use_local_synchronization=True
                            )
                        else:
                            local_ranks = nn_structure[f"sd{sd}"][f"r0"][f"s{s}"][
                                f"sh{sh}"
                            ]["local_ranks"]
                            local_group = nn_structure[f"sd{sd}"][f"r0"][f"s{s}"][
                                f"sh{sh}"
                            ]["local_group"]

                        nn_structure[f"sd{sd}"][f"r{rep}"][f"s{s}"][f"sh{sh}"] = {
                            "global_ranks": global_ranks,
                            "local_ranks": local_ranks,
                            "shard_ranks": ranks_on_this_stage,
                            "rank": rank,
                            "global_group": global_group,
                            "local_group": local_group,
                        }
        return nn_structure

    def _get_stage_list(self):
        stage_list = list(self.organized_layers.values())
        if "start" not in stage_list[0]:
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
        if dist.get_rank() == 0:
            pprint.pprint(
                f"Net dict: {net} \n Number of keys: {len(net.keys())}. Keys: {net.keys()}"
            )

        organized_layers = {}
        dst = net["start"]["dst"]["to"]
        organized_layers[net["start"]["stage"]] = ["start"]
        while dst:
            next_dst = []
            for layer_name in dst:
                organized_layers.setdefault(net[layer_name]["stage"], [])
                if layer_name not in organized_layers[net[layer_name]["stage"]]:
                    organized_layers[net[layer_name]["stage"]].append(layer_name)
                next_dst.extend(net[layer_name]["dst"]["to"])
            dst = next_dst

        dict2 = {}
        counter = 0
        for key, value in organized_layers.items():
            if "start" in value:
                dict2[0] = value
            if "finish" not in value:
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
                result.insert(0, node)

        for node in graph:
            if node not in visited:
                visit(node)

        return result

    def _validate_network(self):
        net = copy.deepcopy(self.net_dict)
        if "start" not in net:
            raise ValueError("Network must have a layer called 'start'.")
        if "finish" not in net:
            raise ValueError("Network must have a layer called 'finish'.")

        for k, v in net.items():
            v["stage"] = 1

        for layer_name in net.keys():
            if not layer_name.isidentifier():
                raise ValueError(f"Layer '{layer_name}' contains invalid characters.")

        errors = []
        for layer_name, layer_info in net.items():
            if "rcv" not in layer_info:
                errors.append(f"Layer '{layer_name}' is missing 'rcv' entry.")
                continue
            if "dst" not in layer_info:
                errors.append(f"Layer '{layer_name}' is missing 'dst' entry.")
                continue

            rcv_sources = layer_info["rcv"].get("src", [])
            rcv_strategy = layer_info["rcv"].get("strategy")
            if rcv_sources is None:
                rcv_sources = []
            else:
                if len(rcv_sources) > 1 and rcv_strategy is None:
                    errors.append(
                        f"Layer '{layer_name}' has multiple receive sources but no strategy."
                    )

            for src in rcv_sources:
                if src not in net:
                    errors.append(
                        f"Layer '{layer_name}' has 'rcv' source '{src}' which does not exist."
                    )

            dst_targets = layer_info["dst"].get("to", [])
            if dst_targets is None:
                dst_targets = []
            for dst in dst_targets:
                if dst not in net:
                    errors.append(
                        f"Layer '{layer_name}' has 'dst' target '{dst}' which does not exist."
                    )

            for dst in dst_targets:
                dst_rcv_sources = net[dst]["rcv"].get("src", [])
                if layer_name not in dst_rcv_sources:
                    errors.append(
                        f"Layer '{layer_name}' lists '{dst}' as a destination, but '{dst}' does not list it as a source."
                    )

            for src in rcv_sources:
                src_dst_targets = net[src]["dst"].get("to", [])
                if layer_name not in src_dst_targets:
                    errors.append(
                        f"Layer '{layer_name}' lists '{src}' as a source, but '{src}' does not list it as a destination."
                    )

        stages = {}
        for layer_name, layer_info in net.items():
            stage = layer_info.get("stage")
            if stage is None:
                errors.append(f"Layer '{layer_name}' is missing 'stage' entry.")
                continue
            stages.setdefault(stage, []).append(layer_name)

        for stage, layers in stages.items():
            graph = {layer: [] for layer in layers}
            for layer in layers:
                layer_info = net[layer]
                rcv_sources = layer_info["rcv"].get("src", [])
                if rcv_sources is None:
                    rcv_sources = []
                for src in rcv_sources:
                    if src in layers:
                        graph[src].append(layer)
            try:
                self._topological_sort(graph)
            except Exception as e:
                errors.append(f"Cycle detected in stage {stage}: {e}")

        if errors:
            raise ValueError("Network validation failed:\n" + "\n".join(errors))
