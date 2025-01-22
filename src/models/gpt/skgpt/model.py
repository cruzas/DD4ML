from collections import deque

from src.models.gpt.base_gpt import *


class StartLayer(nn.Module):
    """Initial layer for the GPT model"""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size
    
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        pos = pos.unsqueeze(0).expand(b, t)  # shape (b, t)
        tok_emb = self.wte(idx)  # token embeddings
        pos_emb = self.wpe(pos)  # position embeddings
        x = self.drop(tok_emb + pos_emb)
        return x


class GPT(BaseGPT):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = BaseGPT.get_default_config()
        # pipelining options
        C.num_stages = 1
        C.num_subdomains = 1
        C.num_replicas_per_subdomain = 1
        return C

    def __init__(self, config):
        super().__init__(config)
        self.model_dict = self.build_gpt_dictionary(config)

    def __str__(self):
        dict_str = ""
        for key in self.model_dict:
            dict_str += f"{key}: {self.model_dict[key]['stage']}\n"
        return dict_str

    def build_gpt_dictionary(self, config):
        model_dict = {}

        # Word Embeddings (receives token idx)
        model_dict['start'] = {
            'callable': {
                'object': StartLayer,
                'settings': {'config': config}
            },
            'dst': {'to': ['block_0']},
            'rcv': {'src': [], 'strategy': None},
            'stage': 0,
            'num_layer_shards': 1,
        }

        # Transformer Blocks
        for i in range(config.n_layer):
            next_module = f'block_{i+1}' if i + 1 < config.n_layer else 'ln_f'
            prev_module = f'block_{i-1}' if i > 0 else 'start'
            model_dict[f'block_{i}'] = {
                'callable': {
                    'object': Block,
                    'settings': {'config': config}
                },
                'dst': {'to': [next_module]},
                'rcv': {'src': [prev_module], 'strategy': None},
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

        # Optionally reset all stages (if your pipeline approach needs it)
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

    def forward(self, x):
        # Implemented in parallelized model
        pass

    def generate(self, x):
        # Implemented in parallelized model
        pass