import os

import torch
import torch.distributed as dist
import torch.nn as nn

import dd4ml.utility as utils


class BasePMWModel(nn.Module):
    n_layer = None

    def __init__(self):
        super().__init__()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()
        self.backend = dist.get_backend()

        # Single device computation to avoid redundancy
        # Add safety guard for CUDA device initialization in multiprocessing
        try:
            self.device = (
                torch.device(f"cuda:{self.local_rank}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            # Test device accessibility
            if self.device.type == "cuda":
                torch.tensor([0.0], device=self.device)
        except (RuntimeError, AssertionError) as e:
            print(f"Warning: CUDA device initialization failed, falling back to CPU: {e}")
            self.device = torch.device("cpu")
        # Maintain backward compatibility
        self.tensor_device = self.device
        self.default_device = self.device

        # Cache for efficiency
        self._rank_lookup_cache = {}
        self._layer_stage_cache = {}
        self._is_cache_valid = False

    def backend_device(self, tensor=torch.tensor([0])):
        backend_device = torch.device("cpu")
        if torch.cuda.is_available() and dist.get_backend() == "nccl":
            backend_device = (
                tensor.device
                if "cuda" in str(tensor.device)
                else torch.device(f"cuda:{self.local_rank}")
            )
        return backend_device

    def _setup_single_rank_structure(self):
        """Simplified setup for single rank case."""
        self.all_model_ranks = [[[0]]]
        self.all_model_ranks_flat = [0]
        self.layer_copies = {"stage0_shard0": [0]}
        self.last_layers_main_shard = [0]
        self.subdomain_ranks = [0]
        self.all_final_stages_main_rank = [0]
        self._rank_lookup_cache = {0: {
            "subdomain_idx": 0, "replica_idx": 0, "stage_idx": 0,
            "subdomain_ranks": [[0]], "replica_ranks": [[0]], "stage_ranks": [0]
        }}
        self._is_cache_valid = True

    def distributed_model_rank_structure(
        self,
        subdomains,
        replicas_per_subdomain,
        stages_amount,
        gpus_per_sharded_layer,
        node_rank_dict,
    ):
        # Early return for single process case
        if self.world_size == 1:
            simple_ranks = [[[0]]]
            self._setup_single_rank_structure()
            return simple_ranks
        """
        Outputs a list e.g., if we have
            - 2 subdomains in data (Domain Decomposition in data)
            - 2 replicas per subdomain (data parallel within a subdomain)
            - 2 stages (pipeline)
            - 2 GPUs per stage (layer sharding - tensor parallelism) <- for improved performance it would be better to use this only in case multiple GPUs are available per node
        the output will be:
        [
            [ # Subdomain 0
                [ # Replica 0
                    [ # Stage 0
                        [0, 1], # Stage 0 is sharded across 2 GPUs
                    ],
                    [ # Stage 1
                        [2, 3], # Stage 1 is sharded across 2 GPUs
                    ],
                ],
                [ # Replica 1
                    [ # Stage 0
                        [4, 5], # Stage 0 is sharded across 2 GPUs
                    ],
                    [ # Stage 1
                        [6, 7], # Stage 1 is sharded across 2 GPUs
                    ],
                ]
            ],

            [ # Subdomain 1 (similar to Subdomain 0)
                ...
            ]
        ]

        """
        gpus_per_rank = utils.check_gpus_per_rank()
        if gpus_per_sharded_layer > gpus_per_rank:
            raise ValueError(
                "The number of GPUs per sharded layer cannot be greater than the number of GPUs per rank."
            )
        if gpus_per_rank % gpus_per_sharded_layer == 0:
            # If the number of GPUs per rank is divisible by the number of GPUs per sharded layer, take all ranks from each node. This will not cause sharding across different nodes.
            rank_list = []
            for node in node_rank_dict:
                for i in node_rank_dict[node]:
                    rank_list.append(i)
        else:
            # Taking all the nodes may lead to sharding across different nodes. To avoid this, we take only the ranks from the first node.
            rank_list = []
            for node in node_rank_dict:
                for i in node_rank_dict[node][:gpus_per_sharded_layer]:
                    rank_list.append(i)
            if (
                subdomains
                * replicas_per_subdomain
                * gpus_per_sharded_layer
                * stages_amount
                > len(rank_list)
            ):
                # If the amount of ranks is not large enough to run the model, take all the ranks from the first node.
                rank_list = []
                for node in node_rank_dict:
                    for i in node_rank_dict[node]:
                        rank_list.append(i)
                if self.rank == rank_list[0]:
                    print(
                        "Not enough GPUs per node to run the model. Taking all ranks from all nodes. NOTE that this may lead to sharding across different nodes, hence poor performance."
                    )
        ranks = []
        c = -1
        layer_copies = {}
        subdomain_final_stages_main_rank = [
            [] for _ in range(subdomains)
        ]  # Fix mutable default
        all_final_stages_main_rank = []

        # Pre-compute stage strings for efficiency
        stage_strings = {
            (i, j): f"stage{i}_shard{j}"
            for i in range(stages_amount)
            for j in range(gpus_per_sharded_layer)
        }
        final_stage_key = f"stage{stages_amount - 1}_shard0"
        for sd in range(subdomains):
            subdomain_ranks = []
            for r in range(replicas_per_subdomain):
                replica_ranks = []
                for i in range(stages_amount):
                    # Pre-allocate list with known size for better memory efficiency
                    stage = [None] * gpus_per_sharded_layer
                    for j in range(gpus_per_sharded_layer):
                        c += 1
                        current_rank = rank_list[c]
                        stage[j] = current_rank

                        # Use pre-computed string
                        stage_key = stage_strings[(i, j)]
                        if stage_key not in layer_copies:
                            layer_copies[stage_key] = [current_rank]
                        else:
                            layer_copies[stage_key].append(current_rank)

                        # Only check once for final stage main shard
                        if i == stages_amount - 1 and j == 0:
                            subdomain_final_stages_main_rank[sd].append(current_rank)
                            all_final_stages_main_rank.append(current_rank)
                    replica_ranks.append(stage)
                subdomain_ranks.append(replica_ranks)
            ranks.append(subdomain_ranks)
        self.layer_copies = layer_copies
        self.all_model_ranks = ranks
        self.all_model_ranks_flat = utils.list_flattener(ranks)
        self.all_model_ranks_group = dist.new_group(
            ranks=self.all_model_ranks_flat, use_local_synchronization=True
        )

        # Store in each rank the correct layer_copies field - this will be needed to synchronize the parameters across the replicas
        # list of last layers main shards
        self.last_layers_main_shard = layer_copies[final_stage_key]
        self.all_layer_copies_group = None
        for layer in layer_copies:
            # last layers and main shard (0) are responsible for the computation of the loss
            if final_stage_key == layer:
                self.last_layers_main_shard_group = dist.new_group(
                    ranks=self.last_layers_main_shard, use_local_synchronization=True
                )
            if self.rank in layer_copies[layer]:
                self.all_layer_copies = layer_copies[layer]
                self.all_layer_copies_group = dist.new_group(
                    ranks=self.all_layer_copies, use_local_synchronization=True
                )
                break

        # create subdomain groups
        # first replica, last stage, first GPU (main shard)
        self.last_layer_main_shard = self.subdomain_rank_structure()[0][-1][0]
        self.subdomain_ranks = self.subdomain_rank_structure(flatten=True)
        self.subdomain_ranks_group = dist.new_group(
            ranks=self.subdomain_rank_structure(flatten=True),
            use_local_synchronization=True,
        )
        self.subdomain_final_stages_main_rank = [None]
        self.subdomain_final_stages_main_rank_group = None
        for sd in range(subdomains):
            if self.rank in subdomain_final_stages_main_rank[sd]:
                self.subdomain_final_stages_main_rank = (
                    subdomain_final_stages_main_rank[sd]
                )
                self.subdomain_final_stages_main_rank_group = dist.new_group(
                    ranks=self.subdomain_final_stages_main_rank,
                    use_local_synchronization=True,
                )
                break
        self.replicas_in_subdomain_final_stages_main_rank = [
            r
            for r in self.subdomain_final_stages_main_rank
            if r in self.subdomain_ranks
        ]
        self.replicas_in_subdomain_final_stages_main_rank_group = dist.new_group(
            ranks=self.replicas_in_subdomain_final_stages_main_rank,
            use_local_synchronization=True,
        )
        # create global group of final stages main rank
        self.all_final_stages_main_rank = all_final_stages_main_rank
        self.all_final_stages_main_rank_group = dist.new_group(
            ranks=self.all_final_stages_main_rank, use_local_synchronization=True
        )

        # Build lookup table for efficient rank structure queries
        self._build_rank_lookup_table()
        return ranks

    def _build_rank_lookup_table(self):
        """Build a lookup table to optimize rank structure queries."""
        for sd_idx, subdomain_ranks in enumerate(self.all_model_ranks):
            for r_idx, replica_ranks in enumerate(subdomain_ranks):
                for s_idx, stage_ranks in enumerate(replica_ranks):
                    for rank in stage_ranks:
                        self._rank_lookup_cache[rank] = {
                            "subdomain_idx": sd_idx,
                            "replica_idx": r_idx,
                            "stage_idx": s_idx,
                            "subdomain_ranks": subdomain_ranks,
                            "replica_ranks": replica_ranks,
                            "stage_ranks": stage_ranks,
                        }
        self._is_cache_valid = True

    def from_rank_structure_to_layer_number(self):
        """
        This function uses "self.all_model_ranks" and "self.rank" to return the layer number corresponding to the current rank.
        """
        if self._is_cache_valid:
            cached_info = self._rank_lookup_cache.get(self.rank)
            if cached_info:
                return cached_info["stage_idx"]

        # Fallback to original method if cache not available
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                # replica_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]], e.g. stage_ranks = [0, 1]
                for stage_ranks in replica_ranks:
                    if self.rank in stage_ranks:  # sharded tensor
                        return replica_ranks.index(stage_ranks)

    def subdomain_rank_structure(self, flatten=False):
        """
        This function returns the rank structure of the subdomains which contains the current self.rank.
        """
        if self._is_cache_valid:
            cached_info = self._rank_lookup_cache.get(self.rank)
            if cached_info:
                subdomain_ranks = cached_info["subdomain_ranks"]
                return utils.list_flattener(subdomain_ranks) if flatten else subdomain_ranks

        # Fallback to original method
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                for stage_ranks in replica_ranks:
                    if self.rank in stage_ranks:
                        if flatten:
                            return utils.list_flattener(subdomain_ranks)
                        else:
                            return subdomain_ranks
        return None

    def replica_rank_structure(self):
        """
        This function returns the rank structure of the replicas which contains the current self.rank.
        """
        if self._is_cache_valid:
            cached_info = self._rank_lookup_cache.get(self.rank)
            if cached_info:
                return cached_info["replica_ranks"]

        # Fallback to original method
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                for stage_ranks in replica_ranks:
                    if self.rank in stage_ranks:
                        return replica_ranks
        return None

    def stage_rank_structure(self):
        """
        This function returns the rank structure of the stages which contains the current self.rank.
        """
        if self._is_cache_valid:
            cached_info = self._rank_lookup_cache.get(self.rank)
            if cached_info:
                return cached_info["stage_ranks"]

        # Fallback to original method
        for subdomain_ranks in self.all_model_ranks:
            for replica_ranks in subdomain_ranks:
                for stage_ranks in replica_ranks:
                    if self.rank in stage_ranks:
                        return stage_ranks
        return None
