from dataclasses import dataclass
import math 
import torch 
import torch.nn.functional as F
import torch.nn as nn

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

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

class Block(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = None  # Will set this later based on tokenizer
    n_layer: int = 6        # Reduced layers for faster training
    n_head: int = 6         # Reduced heads
    n_embd: int = 384       # Reduced embedding size
    dropout: float = 0.2
    bias: bool = True       # Use bias in Linear and LayerNorm layers

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

# -----------------------------
# Model Dictionary Definition
# -----------------------------

def get_model_dict(config):
    model = {}

    # Start layer (embedding and positional encoding)
    model['start'] = {
        'callable': {'object': StartLayer, 'settings': {'config': config}},
        'dst': {'to': ['block_0']},
        'rcv': {'src': [], 'strategy': None},
        'stage': 1,
        'num_layer_shards': 1,
    }

    # Transformer blocks
    for i in range(config.n_layer):
        model[f'block_{i}'] = {
            'callable': {'object': Block, 'settings': {'config': config}},
            'dst': {'to': [f'block_{i+1}'] if i+1 < config.n_layer else ['ln_f']},
            'rcv': {'src': [f'block_{i-1}'] if i > 0 else ['start'], 'strategy': None},
            'stage': 1,
            'num_layer_shards': 1,
        }

    # Final LayerNorm layer
    model['ln_f'] = {
        'callable': {'object': LNFLayer, 'settings': {'config': config}},
        'dst': {'to': ['lm_head']},
        'rcv': {'src': [f'block_{config.n_layer - 1}'], 'strategy': None},
        'stage': 1,
        'num_layer_shards': 1,
    }

    # Language Modeling Head
    model['lm_head'] = {
        'callable': {'object': LMHeadLayer, 'settings': {'config': config}},
        'dst': {'to': []},
        'rcv': {'src': ['ln_f'], 'strategy': None},
        'stage': 1,
        'num_layer_shards': 1,
    }

    return model