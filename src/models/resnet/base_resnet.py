from abc import abstractmethod

import torch.nn as nn

from src.models.base_model import *
from src.utils import CfgNode as CN


class BaseResNet(BaseModel):
    """Abstract Base Class for ResNet models."""

    @staticmethod
    def get_default_config():
        C = BaseModel.get_default_config()
        C.model_type = 'resnet18'
        C.input_channels = None   # Number of input channels (e.g., 3 for RGB images)
        C.output_classes = None   # Number of output classes (e.g., 10 for CIFAR-10)
        C.block_layers = None     # List of integers specifying the number of blocks per stage
        C.num_filters = 64        # Initial number of filters
        C.dropout_p = 0.1         # Dropout probability for fully connected layers
        return C

    def __init__(self, config):
        super().__init__(config)
        assert config.input_channels is not None
        assert config.output_classes is not None

        type_given = config.model_type is not None
        params_given = config.block_layers is not None
        assert type_given ^ params_given  # exactly one of these (XOR)

        if type_given:
            # Translate model_type into detailed layer configurations
            config.merge_from_dict({
                'resnet18': {
                    'block_layers': [2, 2, 2, 2],
                    'num_filters': 64,
                },
                'resnet34': {
                    'block_layers': [3, 4, 6, 3],
                    'num_filters': 64,
                },
                # Additional ResNet variants can be added here.
            }[config.model_type])

        self.input_channels = config.input_channels
        self.output_classes = config.output_classes
        self.block_layers = config.block_layers
        self.num_filters = config.num_filters
        self.dropout_p = config.dropout_p

    @abstractmethod
    def forward(self, x):
        """Define the forward pass."""
        pass
