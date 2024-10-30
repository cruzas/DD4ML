from dataclasses import dataclass
import math 
import torch 
import torch.nn.functional as F
import torch.nn as nn

@dataclass
class GPTConfig:
    num_stages: int = 10
    block_size: int = 256
    vocab_size: int = None  # Will set this later based on tokenizer
    n_layer: int = 6        # Reduced layers for faster training
    n_head: int = 6         # Reduced heads
    n_embd: int = 384       # Reduced embedding size
    dropout: float = 0.2
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
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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
        x = data[0]
        head_outputs = data[1:]
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
        self.c_attn = nn.Linear(config.n_embd, 3 * self.head_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
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
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # shape: (B, T, T)
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
    tot_layers = len(net_dict)
    
    layers_per_stage = tot_layers // max_stages

    
    dst = net_dict['start']['dst']['to']
    stage = 0
    net_dict['start']['stage'] = 0
    layers_per_stage[0] -= 1
    
    while dst:
        next_dst = []
        for layer_name in dst:
            if layer_name not in organized_layers[net_dict[layer_name]['stage']]:
                organized_layers[net_dict[layer_name]['stage']].append(layer_name)
            next_dst.extend(net_dict[layer_name]['dst']['to'])
        dst = next_dst

    return net_dict

def get_model_dict(config):
    model = {}

    n_layer = config.n_layer
    tot_stages=config.num_stages # TODO: this is for debugging. Put somewhere else.
    
    # ----------------------------------------- Model Layers -----------------------------------------
    # Start layer (embedding and positional encoding)
    layer_idx = 0
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
        layer_idx = blk_idx*3
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
                'stage': 0,
                'num_layer_shards': 1,
            }

        # Combine heads
        layer_idx = 1+blk_idx*3
        model[f'block_{blk_idx}_combine_heads'] = {
            'callable': {'object': CombineHeadsBlock, 'settings': {'config': config}},
            'dst': {'to': [f'block_{blk_idx}_partC']},
            'rcv': {'src': [f'block_{blk_idx}_partA'] + [f'block_{blk_idx}_head_{head_idx}' for head_idx in range(config.n_head)], 'strategy': combine_heads},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # LayerNormAndMLPBlock
        layer_idx = 2+blk_idx*3
        model[f'block_{blk_idx}_partC'] = {
            'callable': {'object': LayerNormAndMLPBlock, 'settings': {'config': config}},
            'dst': {'to': [f'block_{blk_idx+1}_partA'] if blk_idx + 1 < config.n_layer else ['ln_f']},
            'rcv': {'src': [f'block_{blk_idx}_combine_heads'], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

    # Final LayerNorm layer
    layer_idx = 1
    model['ln_f'] = {
        'callable': {'object': LNFLayer, 'settings': {'config': config}},
        'dst': {'to': ['finish']},
        'rcv': {'src': [f'block_{config.n_layer - 1}_partC'], 'strategy': None},
        'stage': 0,
        'num_layer_shards': 1,
    }

    # Language Modeling Head
    layer_idx = 2
    model['finish'] = {
        'callable': {'object': LMHeadLayer, 'settings': {'config': config}},
        'dst': {'to': []},
        'rcv': {'src': ['ln_f'], 'strategy': None},
        'stage': 0,
        'num_layer_shards': 1,
    }
    
    return model



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
#     config = GPTConfig(
#         block_size=256,
#         vocab_size=2,
#         n_layer=6,
#         n_head=6,
#         n_embd=384,
#         dropout=0.2,
#         bias=True
#     )
#     a = get_model_dict(config)
    
#     # compute the amount of different stages in the model
#     stages = set()
#     for key in a.keys():
#         stages.add(a[key]['stage'])
    
#     print(stages)
    
#     # create a model from the dictionary
    
#     model = GPTModelFromDict(a)
    
#     print('asd')