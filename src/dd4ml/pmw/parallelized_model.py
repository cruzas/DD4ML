import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .base_pmw_model import BasePMWModel
from .data_and_weight_parallelized_subdomain import \
    DataAndWeightParallelizedSubdomain


class ParallelizedModel(BasePMWModel):
    def __init__(self, model_handler, sample):
        """
        Ranks that will be used are [0, ..., world_size - 1]

        This is the outermost shell in a multi-shell parallel strategy.
        num_subdomains is the next shell, which deals with data-parallelism (Domain Decomposition approach).
        num_replicas_per_subdomain refers to the number of replicas in each data-parallel subdomain (exact data parallelism to speed up computation within each subdomain).
        stage_list is the list of pipeline stages per replica in each subdomain

        NOTE: REMAKE COMMENTS !!!!!!!!!!!!!!!!!!!

        E.g. num_subdomains = 2; num_replicas_per_subdomain = 3; stage_list = [(Layer0, Layer0dict), (Layer1, Layer1dict), (Layer2, Layer2dict)])

                                    Subdomain 0                                                         Subdomain 1
                Replica 0           Replica 1           Replica 2                   Replica 0           Replica 1           Replica 2
                [Layer0, (Rank0)    [Layer0, (Rank3)   [Layer0, (Rank6)            [Layer0, (Rank9)    [Layer0, (Rank12)   [Layer0, (Rank15)
                 Layer1, (Rank1)     Layer1, (Rank4)    Layer1, (Rank7)             Layer1, (Rank10)    Layer1, (Rank13)    Layer1, (Rank16)
                 Layer2] (Rank2)     Layer2] (Rank5)    Layer2] (Rank8)             Layer2] (Rank11)    Layer2] (Rank14)    Layer2] (Rank17)
        """
        super().__init__()

        self.model_handler = model_handler
        self.num_subdomains = self.model_handler.num_subdomains
        self.num_replicas_per_subdomain = self.model_handler.num_replicas_per_subdomain
        self.tot_replicas = self.model_handler.tot_replicas
        self.block_size = sample.size(1)

        if self.rank in self.model_handler.available_ranks:
            self.subdomain = DataAndWeightParallelizedSubdomain(
                self.model_handler, sample
            )
            self.sync_params()

    def save_state_dict(self, path):
        params_dict = self.state_dict(dst_rank=0)
        if self.rank == 0:
            torch.save(params_dict, path)
            print("Saved par model to: ", path)

    def load_state_dict(self, path):
        params_dict = torch.load(path)
        local_layers = self.model_handler.stage_data()["layers"]
        local_dict = {}
        for layer in local_layers:
            for key in params_dict.keys():
                if layer in key:
                    if layer not in local_dict:
                        local_dict[layer] = {}
                    key2 = ".".join(key.split(".")[1:])
                    key2 = "layer." + key2
                    local_dict[layer][key2] = params_dict[key]
        self.subdomain.weight_parallelized_model.subdomain.load_state_dict(local_dict)

    def state_dict(self, dst_rank=0):
        def merger(gathered_state_dicts, replica_ranks):
            # Merge the gathered state_dicts
            for i in range(1, len(replica_ranks)):
                for key in gathered_state_dicts[i].keys():
                    gathered_state_dicts[0][key] = gathered_state_dicts[i][key]
            return gathered_state_dicts[0]

        sd, rep, _, _ = self.model_handler.rank_to_position()
        if sd == 0 and rep == 0:
            replica_ranks = self.model_handler.replica_ranks()
            subdomain_state_dict = (
                self.subdomain.weight_parallelized_model.subdomain.state_dict()
            )
            if self.rank == replica_ranks[0]:
                gathered_state_dicts = [None for _ in range(len(replica_ranks))]
                dist.gather_object(
                    subdomain_state_dict,
                    gathered_state_dicts,
                    dst=replica_ranks[0],
                    group=self.model_handler.get_replica_group(),
                )
            else:
                dist.gather_object(
                    subdomain_state_dict,
                    dst=replica_ranks[0],
                    group=self.model_handler.get_replica_group(),
                )

            if self.rank == replica_ranks[0]:
                merged_state_dict = merger(gathered_state_dicts, replica_ranks)
                if replica_ranks[0] != dst_rank:
                    dist.send_object_list([merged_state_dict], dst=dst_rank)
                else:
                    return merged_state_dict

        if self.rank == dst_rank:
            return dist.recv_object_list(src=replica_ranks[0])[0]

    def parameters(self):
        return self.subdomain.weight_parallelized_model.subdomain.parameters()

    def configure_params(self, train_config):
        return self.subdomain.weight_parallelized_model.subdomain.configure_params(
            train_config
        )

    def subdomain_forward(self):
        return self.subdomain.weight_parallelized_model.subdomain.forward()

    def subdomain_backward(self, losses):
        self.subdomain.weight_parallelized_model.subdomain.backward(losses)
        # This is needed in case there are multiple replicas per subdomain (exact data parallelism)
        self.subdomain.sync_grads()

    def subdomain_zero_grad(self):
        self.subdomain.weight_parallelized_model.subdomain.zero_grad()

    def subdomain_params(self):
        return self.subdomain.weight_parallelized_model.subdomain.parameters()

    def subdomain_grad(self):
        return self.subdomain.weight_parallelized_model.subdomain.grad()

    def subdomain_grad_norm(self, p=2):
        return self.subdomain.weight_parallelized_model.subdomain.grad_norm(p=p)

    def parameters(self, clone=True):  # Returns the global parameters of the model
        return self.subdomain.weight_parallelized_model.parameters(clone=clone)

    def parameters_norm(self, p=2):  # Returns the global parameters norm of the model
        return self.subdomain.weight_parallelized_model.parameters().norm(p=p)

    def grad(self, clone=True):  # Returns the global gradient of the model
        return self.subdomain.weight_parallelized_model.grad(clone=clone)

    def grad_norm(self, p=2):  # Returns the global gradient norm of the model
        return self.subdomain.weight_parallelized_model.grad_norm(p=p)

    def forward(self, x, chunks_amount=1, reset_grad=False, compute_grad=True):
        return self.subdomain.forward(
            x,
            chunks_amount=chunks_amount,
            reset_grad=reset_grad,
            compute_grad=compute_grad,
        )

    def backward(self, losses):
        self.subdomain.backward(losses=losses, sync=False)
        self.sync_grads()

    def sync_params(self, method="average"):
        if self.num_subdomains > 1:
            if method not in ["average", "sum"]:
                raise ValueError(f"Method {method} is not supported.")
            for param in self.subdomain_params():
                dist.all_reduce(
                    tensor=param.data,
                    group=self.model_handler.get_layers_copy_group(mode="global"),
                    op=dist.ReduceOp.SUM,
                )
                if method == "average":
                    param.data /= self.tot_replicas

    def sync_grads(self):
        if self.num_subdomains > 1:
            for param in self.subdomain_params():
                dist.all_reduce(
                    tensor=param.grad,
                    group=self.model_handler.get_layers_copy_group(mode="global"),
                    op=dist.ReduceOp.SUM,
                )
                param.grad /= self.tot_replicas

    def normalize_grads(self, p=torch.inf):
        norm = self.subdomain.weight_parallelized_model.grad_norm(p=p)
        for param in self.subdomain_params():
            param.grad /= norm

    # ------------------ Methods for transformer networks -------------------
    def subdomain_named_params(self):
        # Required for proper parameter initialization for transformer networks
        return self.subdomain.weight_parallelized_model.subdomain.named_parameters()

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        do_sample=False,
        top_k=None,
        closure=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        last_stage_ranks = self.model_handler.get_stage_ranks(
            stage_name="last", mode="local"
        )
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )

            # forward the model to get the logits for the index in the sequence
            logits = self.forward(
                idx_cond, chunks_amount=1, reset_grad=False, compute_grad=False
            )[0]

            if not self.model_handler.is_last_stage():
                # increase second dimension of idx by one to receive it in the broadcast
                idx = torch.cat(
                    (idx, torch.zeros((idx.size(0), 1), dtype=torch.long)), dim=1
                )

            if self.model_handler.is_last_stage():
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # either sample from the distribution or take the most likely element
                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    _, idx_next = torch.topk(probs, k=1, dim=-1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

            # broadcast the idx to all replicas in the last stage
            idx_broadcast = dist.broadcast(
                idx,
                src=last_stage_ranks[0],
                group=self.model_handler.get_replica_group(),
                async_op=True,
            )
            idx_broadcast.wait()
        return idx
