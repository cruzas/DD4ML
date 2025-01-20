"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from collections import deque

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from optimizers.trust_region import TrustRegion
from src.utils import CfgNode as CN
from src.utils import broadcast_dict

# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(
            m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt-mini'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # pipelining options
        C.num_stages = 1
        return C

    def __str__(self):
        dict_str = ""
        for key in self.model_dict:
            dict_str += f"{key}: {self.model_dict[key]['stage']}\n"
        return dict_str

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None,
                           config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                # 117M params
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),
                # GPT-2 configs
                # 124M params
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
                # 350M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
                # 774M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
                # 1558M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.model_dict = self.build_gpt_dictionary(config)

    def build_gpt_dictionary(self, config):
        model_dict = {}

        # Word Embeddings
        model_dict['start'] = {
            'callable': {
                'object': nn.Embedding,
                'settings': {
                    'num_embeddings': config.vocab_size,
                    'embedding_dim': config.n_embd
                }
            },
            'dst': {'to': ['wpe']},
            'rcv': {'src': [], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Positional Embeddings
        model_dict['wpe'] = {
            'callable': {
                'object': nn.Embedding,
                'settings': {
                    'num_embeddings': config.block_size,
                    'embedding_dim': config.n_embd
                }
            },
            'dst': {'to': ['drop']},
            'rcv': {'src': ['start'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Dropout
        model_dict['drop'] = {
            'callable': {
                'object': nn.Dropout,
                'settings': {'p': config.embd_pdrop}
            },
            'dst': {'to': ['block_0']},
            'rcv': {'src': ['wpe'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Transformer Blocks
        for i in range(config.n_layer):
            model_dict[f'block_{i}'] = {
                'callable': {
                    'object': Block,
                    'settings': {'config': config}
                },
                'dst': {
                    'to': (
                        [f'block_{i+1}'] if i + 1 < config.n_layer else ['ln_f']
                    )
                },
                'rcv': {
                    'src': [f'block_{i-1}'] if i > 0 else ['drop'],
                    'strategy': None
                },
                'stage': 0,
                'num_layer_shards': 1,
            }

        # Final LayerNorm
        model_dict['ln_f'] = {
            'callable': {
                'object': nn.LayerNorm,
                'settings': {'normalized_shape': config.n_embd}
            },
            'dst': {'to': ['finish']},
            'rcv': {'src': [f'block_{config.n_layer - 1}'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Language Model Head
        model_dict['finish'] = {
            'callable': {
                'object': nn.Linear,
                'settings': {
                    'in_features': config.n_embd,
                    'out_features': config.vocab_size,
                    'bias': False
                }
            },
            'dst': {'to': []},
            'rcv': {'src': ['ln_f'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Optionally set all stages to 0 again (if needed).
        for name in model_dict:
            model_dict[name]['stage'] = 0

        return self.set_stage(model_dict, config.num_stages)

    def set_stage(self, model_dict, num_stages):
        """
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
