from dataclasses import dataclass
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import utils


@dataclass
class GPTConfig:
    num_stages: int = 10
    block_size: int = 256
    vocab_size: int = None  # Will set this later based on tokenizer
    n_layer: int = 6        # Reduced layers for faster training
    n_head: int = 6         # Reduced heads
    n_embd: int = 384       # Reduced embedding size
    dropout: float = 0.0
    bias: bool = True       # Use bias in Linear and LayerNorm layers


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    """Feed-forward neural network."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNormBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.ln(x)


class AttentionHead(nn.Module):
    def __init__(self, config, head_index):
        super().__init__()
        self.attn = CausalSelfAttention(config, head_index)

    def forward(self, x):
        return self.attn(x)

def combine_heads(*inputs):
    return inputs

class CombineHeadsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    # TODO: make our implementation work with something like this, i.e. with multiple inputs
    def forward(self, data):
        x = data[-1]
        head_outputs = data[0:-1]
        # Concatenate outputs from all attention heads
        y = torch.cat(head_outputs, dim=-1)  # shape: (B, T, n_embd)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        # Residual connection
        x = x + y
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention with single head."""

    def __init__(self, config, head_index):
        super().__init__()
        self.head_dim = config.n_embd // config.n_head
        self.head_index = head_index
        self.c_attn = nn.Linear(
            config.n_embd, 3 * self.head_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .unsqueeze(0))  # shape (1, block_size, block_size)

    def forward(self, x):
        B, T, C = x.size()
        # Compute query, key, and value vectors for this head
        qkv = self.c_attn(x)  # shape (B, T, 3 * head_dim)
        q, k, v = qkv.chunk(3, dim=2)  # shapes: (B, T, head_dim)
        if self.flash:
            # Use flash attention if available
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * \
                (1.0 / math.sqrt(self.head_dim))  # shape: (B, T, T)
            # Apply causal mask
            att = att.masked_fill(self.mask[:, :T, :T] == 0, float('-inf'))
            # Apply softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # Compute attention output
            y = att @ v  # shape: (B, T, head_dim)
        # Apply residual dropout
        y = self.resid_dropout(y)  # shape: (B, T, head_dim)
        return y  # Output is (B, T, head_dim)


class LayerNormAndMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mlp(self.ln(x))
        return x


class StartLayer(nn.Module):
    """Initial layer that applies token and positional embeddings and dropout."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        pos = pos.unsqueeze(0).expand(b, t)  # shape (b, t)
        tok_emb = self.wte(idx)  # token embeddings
        pos_emb = self.wpe(pos)  # position embeddings
        x = self.drop(tok_emb + pos_emb)
        return x


class LNFLayer(nn.Module):
    """Final LayerNorm layer."""

    def __init__(self, config):
        super().__init__()
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.ln_f(x)
        return x


class LMHeadLayer(nn.Module):
    """Language Modeling Head that projects embeddings to vocabulary size."""

    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        logits = self.lm_head(x)
        return logits


def set_stage(net_dict, max_stages):
    '''
    Randomly assign stages to the layers in the model following the path such that the stages only have connected layers.
    '''
    # original_net_dict = copy.deepcopy(net_dict)
    net3 = {}
    if dist.get_rank() == 0:
        print(f"len dict: {len(net_dict)}, amount of stages {max_stages}")
    if max_stages == len(net_dict):
        # every layer on a different stage
        if dist.get_rank() == 0:
            print("Every layer on a different stage.")
        for i, key in enumerate(net_dict.keys()): 
            net_dict[key]['stage'] = i
        return net_dict
    elif max_stages == 1:
        # all layers on the same stage
        if dist.get_rank() == 0:
            print("All layers on the same stage.")
        for key in net_dict.keys(): net_dict[key]['stage'] = 0
        return net_dict
    
    if dist.get_rank() == 0:
        if max_stages < 1: raise ValueError('Number of stages should be at least 1')
        if max_stages > len(net_dict): raise ValueError('Number of stages should be less than the number of layers in the model')
            
        for key in net_dict.keys(): net_dict[key]['stage'] = None if max_stages > 1 else 0

        tot_layers = len(net_dict)
        layers_per_stage = [tot_layers // max_stages] * max_stages
        if tot_layers % max_stages != 0:
            for i in range(tot_layers % max_stages):
                layers_per_stage[i] += 1

        stage_idx = 0
        def _get_available_layer_closest_to_start(net):
            '''
            Follows the flow of the model and returns the next layer that can be assigned a stage.
            '''        
            if net['start']['stage'] is None:
                return 'start'
            dst = net['start']['dst']['to']
            while dst:
                next_dst = []
                for layer_name in dst:
                    if net[layer_name]['stage'] is None:
                        return layer_name
                    next_dst.extend(net[layer_name]['dst']['to'])
                dst = next_dst
        c = 0
        while stage_idx != max_stages:
            c += 1
            if c % 100 == 0:
                layers_per_stage = [layers_per_stage[i] + 1 for i in range(max_stages)]
            stage_idx = 0
            for key in net_dict.keys(): net_dict[key]['stage'] = None # setting all stages to None
            while any([net_dict[key]['stage'] is None for key in net_dict.keys()]): # while there are layers that have not been assigned a stage
                closest_layer = _get_available_layer_closest_to_start(net_dict) # get the next layer that can be assigned a stage
                net_dict[closest_layer]['stage'] = stage_idx 
                counter = 1
                while stage_idx < max_stages - 1 and counter < layers_per_stage[stage_idx]: # assign the rest of the layers in the stage
                    # One-level look-ahead to see if there are any free connections, as we still need nodes linked to the same base structure
                    free_connections = []
                    layers_with_same_stage_idx = [layer for layer in net_dict.keys() if net_dict[layer]['stage'] == stage_idx]
                    for layer in layers_with_same_stage_idx:
                        for dst in net_dict[layer]['dst']['to'] + net_dict[layer]['rcv']['src']:
                            if net_dict[dst]['stage'] is None:
                                free_connections.append(dst)
                    if not free_connections:
                        break
                    else: # Choose next layer at random
                        closest_layer = free_connections[torch.randint(0, len(free_connections), (1,)).item()]
                        free_connections.remove(closest_layer)
                        net_dict[closest_layer]['stage'] = stage_idx
                        counter += 1
                stage_idx += 1
    
        net3 = {key: net_dict[key]['stage'] for key in net_dict.keys()}
    
    # TODO: make sure every rank gets the correct dictionary
    net3 = utils.broadcast_dict(d=net3, src=0)
    for key in net_dict.keys(): net_dict[key]['stage'] = net3[key]
    return net_dict

def get_model_dict(config):
    model = {}

    n_layer = config.n_layer
    # TODO: this is for debugging. Put somewhere else.
    tot_stages = config.num_stages

    # TOTAL LAYERS : 3+n_layer*(3+n_head) 
    # ----------------------------------------- Model Layers -----------------------------------------
    # Start layer (embedding and positional encoding)
    model['start'] = {
        'callable': {'object': StartLayer, 'settings': {'config': config}},
        'dst': {'to': ['block_0_partA']},
        'rcv': {'src': [], 'strategy': None},
        'stage': 0,
        'num_layer_shards': 1,
    }

    # Transformer blocks using LayerNormBlock, AttentionHead, LayerNormAndMLPBlock
    for blk_idx in range(config.n_layer):
        # LayerNormBlock
        model[f'block_{blk_idx}_partA'] = {
            'callable': {'object': LayerNormBlock, 'settings': {'config': config}},
            'dst': {'to': [f'block_{blk_idx}_head_{h}' for h in range(config.n_head)] + [f'block_{blk_idx}_combine_heads']},
            'rcv': {'src': [f'block_{blk_idx-1}_partC'] if blk_idx > 0 else ['start'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Attention heads
        for head_idx in range(config.n_head):
            layer_idx = blk_idx*(3+config.n_head)+head_idx+1
            model[f'block_{blk_idx}_head_{head_idx}'] = {
                'callable': {'object': AttentionHead, 'settings': {'config': config, 'head_index': head_idx}},
                'dst': {'to': [f'block_{blk_idx}_combine_heads']},
                'rcv': {'src': [f'block_{blk_idx}_partA'], 'strategy': None},
                'stage': head_idx+1,
                'num_layer_shards': 1,
            }

        # Combine heads
        model[f'block_{blk_idx}_combine_heads'] = {
            'callable': {'object': CombineHeadsBlock, 'settings': {'config': config}},
            'dst': {'to': [f'block_{blk_idx}_partC']},
            'rcv': {'src': [f'block_{blk_idx}_head_{head_idx}' for head_idx in range(config.n_head)]+[f'block_{blk_idx}_partA'], 'strategy': combine_heads},
            'stage': 1,
            'num_layer_shards': 1,
        }

        # LayerNormAndMLPBlock
        model[f'block_{blk_idx}_partC'] = {
            'callable': {'object': LayerNormAndMLPBlock, 'settings': {'config': config}},
            'dst': {'to': [f'block_{blk_idx+1}_partA'] if blk_idx + 1 < config.n_layer else ['ln_f']},
            'rcv': {'src': [f'block_{blk_idx}_combine_heads'], 'strategy': None},
            'stage': 3,
            'num_layer_shards': 1,
        }

    # Final LayerNorm layer
    model['ln_f'] = {
        'callable': {'object': LNFLayer, 'settings': {'config': config}},
        'dst': {'to': ['finish']},
        'rcv': {'src': [f'block_{config.n_layer - 1}_partC'], 'strategy': None},
        'stage': 4,
        'num_layer_shards': 1,
    }

    # Language Modeling Head
    model['finish'] = {
        'callable': {'object': LMHeadLayer, 'settings': {'config': config}},
        'dst': {'to': []},
        'rcv': {'src': ['ln_f'], 'strategy': None},
        'stage': 5,
        'num_layer_shards': 1,
    }
    
    # set every stage to 0
    for key in model.keys():
        model[key]['stage'] = 0

    return set_stage(model, tot_stages)

class GPTModelFromDict(nn.Module):
    def __init__(self, model_dict):
        super().__init__()
        self.model_dict = model_dict
        self.layers = nn.ModuleDict()
        self.build_model()

        # Example of weight tying if needed
        # self.layers['finish'].lm_head.weight = self.layers['start'].wte.weight

    def build_model(self):
        # Instantiate layers
        for name, layer_info in self.model_dict.items():
            callable_obj = layer_info['callable']['object']
            settings = layer_info['callable']['settings']
            self.layers[name] = callable_obj(**settings)

    def forward(self, idx, targets=None):
        layer_outputs = {}
        processed_layers = set()
        layer_queue = ['start']

        while layer_queue:
            # print(layer_queue)
            layer_name = layer_queue.pop(0)
            if layer_name in processed_layers:
                continue

            layer_info = self.model_dict[layer_name]
            layer = self.layers[layer_name]
            src_names = layer_info['rcv']['src']

            # Check if all inputs are ready
            if all(src in layer_outputs for src in src_names):
                # Gather inputs from source layers
                if src_names:
                    inputs = [layer_outputs[src] for src in src_names]
                    # Apply strategy if specified
                    strategy = layer_info['rcv']['strategy']
                    if strategy:
                        x = strategy(*inputs)
                    else:
                        if len(inputs) == 1:
                            x = inputs[0]
                        else:
                            x = inputs  # Pass list of inputs directly to the layer
                else:
                    # For 'start' layer, input is idx
                    x = idx

                # Compute the output of the current layer
                output = layer(x)
                if layer_name in layer_outputs.keys():
                    print('asd')
                layer_outputs[layer_name] = output
                processed_layers.add(layer_name)

                # Add destination layers to the queue
                dst_layers = layer_info['dst']['to']
                for dst in dst_layers:
                    if dst not in processed_layers and dst not in layer_queue:
                        layer_queue.append(dst)
            else:
                # Re-queue the layer if inputs are not ready
                layer_queue.append(layer_name)
            # print the norm of the output
            # try:
            #     print(f'(SEQ) Layer {layer_name}, input norm {torch.norm(torch.tensor(x.clone.detach() ,dtype=torch.float32))}, output norm: {torch.norm(layer_outputs[layer_name])}, param norm: {torch.norm(next(self.layers[layer_name].parameters()))}')
            # except: # x is tuple so sum it up
            #     x_norm = 0
            #     for i in x:
            #         x_norm += torch.norm(i)
            #     print(f'(SEQ) Layer {layer_name}, input norm {x_norm}, output norm: {torch.norm(layer_outputs[layer_name])}, param norm: {torch.norm(next(self.layers[layer_name].parameters()))}')
        # Retrieve the final output
        logits = layer_outputs['finish']
        return logits












# if __name__ == '__main__':
#     CONFIG = GPTConfig(
#         num_stages=6,
#         block_size=256,
#         vocab_size=0,
#         n_layer=1,
#         n_head=2,
#         n_embd=384,
#         dropout=0.2,
#         bias=True
#     )
#     model = get_model_dict(CONFIG)
    
#     # Example of usage
#     try:
#         ordered_layers = resolve_layer_dependencies(model)
#         print("Model connections are set up correctly.")
#         print("Order of layers for execution:", ordered_layers)
#     except ValueError as e:
#         print("Error in model connections:", e)