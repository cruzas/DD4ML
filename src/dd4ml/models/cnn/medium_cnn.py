import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_cnn import BaseCNN


class MediumCNN(BaseCNN):
    """
    Flexible CNN architecture for hyperparameter testing.

    Allows easy variation of:
    - num_conv_layers: Number of convolutional layers
    - filters_per_layer: Number of filters in each conv layer (can be list or int)
    - fc_width: Width of the fully connected layer before output
    - dropout_p: Dropout probability

    Similar to MediumFFNN but for CNNs.
    """

    @staticmethod
    def get_default_config():
        C = BaseCNN.get_default_config()
        C.input_channels = 1  # MNIST is grayscale
        C.output_classes = 10
        C.num_conv_layers = 4  # Number of conv layers
        C.filters_per_layer = [32, 64, 128, 256]  # Filters for each conv layer
        C.fc_width = 128  # Width of FC layer before output
        C.kernel_size = 3
        C.padding = 1
        C.stride = 1
        C.pool_every = 2  # Apply pooling every N layers (2 means pool after layer 2, 4, 6, etc.)
        C.dropout_p = 0.5
        return C

    def __init__(self, config):
        super().__init__(config)

        # If filters_per_layer is a single int, replicate it
        if isinstance(config.filters_per_layer, int):
            filters = [config.filters_per_layer] * config.num_conv_layers
        else:
            filters = config.filters_per_layer[:config.num_conv_layers]
            # Pad if needed
            while len(filters) < config.num_conv_layers:
                filters.append(filters[-1] if filters else 32)

        # Build convolutional layers
        in_channels = config.input_channels
        conv_layers = []

        for i, out_channels in enumerate(filters, start=1):
            # Convolutional block
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=config.kernel_size,
                    padding=config.padding,
                    stride=config.stride
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            # Add pooling if this is a pooling layer
            if config.pool_every > 0 and i % config.pool_every == 0:
                layer.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))

            setattr(self, f"conv{i}", layer)
            conv_layers.append((f"conv{i}", layer))
            in_channels = out_channels

        self.conv_layers = conv_layers
        self.num_conv_layers = len(filters)

        # Calculate the size after convolutions for MNIST (28x28 input)
        # Each pooling reduces spatial dimensions by 2
        num_pools = config.num_conv_layers // config.pool_every if config.pool_every > 0 else 0
        spatial_size = 28 // (2 ** num_pools)
        fc_input_size = filters[-1] * spatial_size * spatial_size

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_size, config.fc_width)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout_p)
        self.fc_out = nn.Linear(config.fc_width, config.output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through convolutional layers
        for i in range(1, self.num_conv_layers + 1):
            x = getattr(self, f"conv{i}")(x)

        # Flatten and pass through FC layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc_out(x)

        return F.log_softmax(x, dim=1)

    def as_model_dict(self):
        """
        Create model_dict for domain decomposition.
        Note: This is a simplified version - you may need to adjust
        based on your specific DD4ML requirements.
        """
        cfg = self.config

        # Determine filters list
        if isinstance(cfg.filters_per_layer, int):
            filters = [cfg.filters_per_layer] * cfg.num_conv_layers
        else:
            filters = cfg.filters_per_layer[:cfg.num_conv_layers]
            while len(filters) < cfg.num_conv_layers:
                filters.append(filters[-1] if filters else 32)

        model_dict = {}

        # First conv layer
        in_ch = cfg.input_channels
        for i, out_ch in enumerate(filters, start=1):
            use_pool = (cfg.pool_every > 0 and i % cfg.pool_every == 0)

            stage_name = f"conv{i}"
            next_stage = f"conv{i+1}" if i < len(filters) else "fc1"
            prev_stage = f"conv{i-1}" if i > 1 else []

            model_dict[stage_name] = {
                "callable": {
                    "object": nn.Sequential,
                    "settings": {
                        "modules": [
                            {
                                "object": nn.Conv2d,
                                "settings": {
                                    "in_channels": in_ch,
                                    "out_channels": out_ch,
                                    "kernel_size": cfg.kernel_size,
                                    "padding": cfg.padding,
                                    "stride": cfg.stride
                                }
                            },
                            {
                                "object": nn.BatchNorm2d,
                                "settings": {"num_features": out_ch}
                            },
                            {
                                "object": nn.ReLU,
                                "settings": {"inplace": True}
                            }
                        ] + ([{
                            "object": nn.MaxPool2d,
                            "settings": {"kernel_size": 2, "stride": 2}
                        }] if use_pool else [])
                    }
                },
                "dst": {"to": [next_stage]},
                "rcv": {"src": [prev_stage] if prev_stage else [], "strategy": None},
                "stage": i - 1,
                "num_layer_shards": 1
            }
            in_ch = out_ch

        # FC layers
        num_pools = cfg.num_conv_layers // cfg.pool_every if cfg.pool_every > 0 else 0
        spatial_size = 28 // (2 ** num_pools)
        fc_input_size = filters[-1] * spatial_size * spatial_size

        model_dict["fc1"] = {
            "callable": {
                "object": nn.Sequential,
                "settings": {
                    "modules": [
                        {"object": nn.Flatten, "settings": {}},
                        {
                            "object": nn.Linear,
                            "settings": {
                                "in_features": fc_input_size,
                                "out_features": cfg.fc_width
                            }
                        },
                        {"object": nn.ReLU, "settings": {"inplace": True}},
                        {"object": nn.Dropout, "settings": {"p": cfg.dropout_p}}
                    ]
                }
            },
            "dst": {"to": ["finish"]},
            "rcv": {"src": [f"conv{cfg.num_conv_layers}"], "strategy": None},
            "stage": cfg.num_conv_layers,
            "num_layer_shards": 1
        }

        model_dict["finish"] = {
            "callable": {
                "object": nn.Linear,
                "settings": {
                    "in_features": cfg.fc_width,
                    "out_features": cfg.output_classes
                }
            },
            "dst": {"to": []},
            "rcv": {"src": ["fc1"], "strategy": None},
            "stage": cfg.num_conv_layers + 1,
            "num_layer_shards": 1
        }

        self.model_dict = model_dict
        self.set_stage()
        return model_dict
