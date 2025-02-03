from abc import abstractmethod

import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.utils import CfgNode as CN


# -----------------------------------------------------------------------------# 
class BaseResNet(BaseModel):
    """Abstract Base Class for ResNet models."""

    @staticmethod
    def get_default_config():
        C = CN()
        # Either model_type or layer details must be given in the config
        C.model_type = 'resnet34'  # Default ResNet variant
        C.input_channels = None  # Number of input channels (e.g., 3 for RGB images)
        C.output_classes = None  # Number of output classes (e.g., 10 for CIFAR-10)
        C.block_type = 'basic'   # Block type: 'basic' or 'bottleneck'
        C.num_blocks = None      # List of integers specifying the number of blocks per layer
        C.num_filters = None     # List of integers specifying the number of filters per layer
        # Dropout hyperparameters
        C.dropout_p = 0.1        # Dropout probability (if used in fully connected layers)
        return C

    def __init__(self, config):
        super().__init__()
        assert config.input_channels is not None
        assert config.output_classes is not None

        type_given = config.model_type is not None
        params_given = config.num_blocks is not None and config.num_filters is not None
        assert type_given ^ params_given  # exactly one of these (XOR)

        if type_given:
            # Translate model_type into detailed layer configurations
            config.merge_from_dict({
                'resnet18': {
                    'block_type': 'basic',
                    'num_blocks': [2, 2, 2, 2],
                    'num_filters': [64, 128, 256, 512],
                },
                'resnet34': {
                    'block_type': 'basic',
                    'num_blocks': [3, 4, 6, 3],
                    'num_filters': [64, 128, 256, 512],
                },
                'resnet50': {
                    'block_type': 'bottleneck',
                    'num_blocks': [3, 4, 6, 3],
                    'num_filters': [64, 128, 256, 512],
                },
                'resnet101': {
                    'block_type': 'bottleneck',
                    'num_blocks': [3, 4, 23, 3],
                    'num_filters': [64, 128, 256, 512],
                },
                'resnet152': {
                    'block_type': 'bottleneck',
                    'num_blocks': [3, 8, 36, 3],
                    'num_filters': [64, 128, 256, 512],
                },
            }[config.model_type])

        self.input_channels = config.input_channels
        self.output_classes = config.output_classes
        self.block_type = config.block_type
        self.num_blocks = config.num_blocks
        self.num_filters = config.num_filters
        self.dropout_p = config.dropout_p

    @abstractmethod
    def forward(self, x):
        """Define the forward pass."""
        pass


# -----------------------------------------------------------------------------#
# BASIC BUILDING BLOCKS
class ConvBNReLU(nn.Module):
    """
    A generic block for a Conv2D -> BatchNorm -> ReLU (+ optional MaxPool).
    Useful for the ResNet 'start' (conv1).
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=7, 
                 stride=2, 
                 padding=3, 
                 pool_kernel=3, 
                 pool_stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, 
                                 stride=pool_stride, 
                                 padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)  # if you want to mimic ResNet's initial maxpool
        return x


class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # Use the correct intermediate variable
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetHead(nn.Module):
    """
    Final layers for ResNet: 
    - AdaptiveAvgPool2d 
    - Flatten 
    - Linear fully-connected layer
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet(BaseResNet):
    """
    ResNet implementation that extends BaseResNet.
    """
    def __init__(self, config):
        super().__init__(config)

        # 1) Stem
        self.stem = ConvBNReLU(in_channels=self.input_channels,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               pool_kernel=3,
                               pool_stride=2)

        # 2) Main body (4 layers)
        self.in_channels = 64
        self.layer1 = self._make_layer(out_channels=self.num_filters[0], num_blocks=self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(out_channels=self.num_filters[1], num_blocks=self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(out_channels=self.num_filters[2], num_blocks=self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(out_channels=self.num_filters[3], num_blocks=self.num_blocks[3], stride=2)

        # 3) Head
        self.head = ResNetHead(in_channels=self.num_filters[3], num_classes=self.output_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Creates a sequential layer of `num_blocks` BasicResBlocks.
        The first block may have stride=2 if we need downsampling.
        """
        layers = []
        # First block in this layer (could be stride=1 or 2)
        layers.append(BasicResBlock(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels

        # Remaining blocks in this layer
        for _ in range(1, num_blocks):
            layers.append(BasicResBlock(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)       # [B, 64, H/4, W/4] after the pool
        x = self.layer1(x)     # [B, 64, ...]
        x = self.layer2(x)     # [B, 128, ...]
        x = self.layer3(x)     # [B, 256, ...]
        x = self.layer4(x)     # [B, 512, ...]
        x = self.head(x)       # [B, num_classes]
        return x


def ResNet18(config=None):
    """
    Factory function for ResNet18.
    """
    if config is None:
        config = BaseResNet.get_default_config()
    config.model_type = 'resnet18'
    return ResNet(config)


def ResNet34(config=None, input_channels=3, output_classes=10):
    """
    Factory function for ResNet34.
    """
    if config is None:
        config = BaseResNet.get_default_config()
    config.model_type = 'resnet34'
    return ResNet(config)


def build_resnet_dictionary(self, config):
    """
    Build a dictionary describing the ResNet architecture, analogous to build_gpt_dictionary.
    Each entry in the dictionary specifies:
      - The class ('object') and its kwargs ('settings')
      - Where it sends its output ('dst') and from whom it receives input ('rcv')
      - The pipeline 'stage' (if you're doing multi-stage pipeline parallelism)
      - How many shards (num_layer_shards) if you shard each layer
    """
    model_dict = {}

    # 1) Stem (ConvBNReLU)
    model_dict['start'] = {
        'callable': {
            'object': ConvBNReLU,
            'settings': {
                'in_channels': config.input_channels,
                'out_channels': 64,
                'kernel_size': 7,
                'stride': 2,
                'padding': 3,
                'pool_kernel': 3,
                'pool_stride': 2
            }
        },
        'dst': {'to': ['layer0_block0']},   # The stem passes output to the first block of layer0
        'rcv': {'src': [], 'strategy': None},  # This is effectively the "start" node
        'stage': 0,
        'num_layer_shards': 1,
    }

    # 2) Build the per-layer blocks
    #    config.num_blocks and config.num_filters should each be lists of length 4 (e.g. [3,4,6,3]).
    current_src = 'start'
    for layer_idx in range(len(config.num_blocks)):
        num_blocks = config.num_blocks[layer_idx]
        out_ch = config.num_filters[layer_idx]
        # For layer 0, the in_channels is 64 (from the stem).
        # For subsequent layers, the in_channels is whatever the previous layer's out_channels was.
        in_ch = 64 if layer_idx == 0 else config.num_filters[layer_idx - 1]

        for block_idx in range(num_blocks):
            block_name = f'layer{layer_idx}_block{block_idx}'

            # If this is the final block in the layer, point 'dst' to the next layer’s first block
            # or to 'finish' if it’s the last layer.
            if block_idx < num_blocks - 1:
                next_module = f'layer{layer_idx}_block{block_idx + 1}'
            else:
                if layer_idx < len(config.num_blocks) - 1:
                    next_module = f'layer{layer_idx + 1}_block0'
                else:
                    next_module = 'finish'

            # Stride is 2 at the first block of each layer (except layer 0)
            stride = 2 if (layer_idx > 0 and block_idx == 0) else 1

            model_dict[block_name] = {
                'callable': {
                    'object': BasicResBlock,
                    'settings': {
                        'in_channels': in_ch,
                        'out_channels': out_ch,
                        'stride': stride
                    }
                },
                'dst': {'to': [next_module]},
                'rcv': {
                    'src': [f'layer{layer_idx}_block{block_idx - 1}']
                           if block_idx > 0
                           else [current_src],
                    'strategy': None
                },
                'stage': 0,
                'num_layer_shards': 1,
            }
            # After the first block in this layer, in_channels = out_ch
            in_ch = out_ch

        # Update the source for the next layer to the final block in this layer
        current_src = f'layer{layer_idx}_block{num_blocks - 1}'

    # 3) Head (AdaptiveAvgPool2d -> Flatten -> Linear)
    model_dict['finish'] = {
        'callable': {
            'object': ResNetHead,
            'settings': {
                'in_channels': config.num_filters[-1],
                'num_classes': config.output_classes
            }
        },
        'dst': {'to': []},  # No further module; this is the end
        'rcv': {'src': [current_src], 'strategy': None},
        'stage': 0,
        'num_layer_shards': 1,
    }

    # Optionally, reset/assign pipeline stages if your code uses them
    for name in model_dict:
        model_dict[name]['stage'] = 0

    # If you have a set_stage(...) helper (like the GPT example), you can call it:
    # return self.set_stage(model_dict, config.num_stages)

    return model_dict
