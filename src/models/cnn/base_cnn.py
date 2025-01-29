from abc import abstractmethod
from collections import deque

from src.models.base_model import *
from src.utils import CfgNode as CN


class BaseCNN(BaseModel):
    """Abstract Base Class for CNN models."""

    @staticmethod
    def get_default_config():
        C = CN()
        # Either model_type or layer details must be given in the config
        C.model_type = 'cnn-small'
        C.input_channels = None  # Number of input channels (e.g., 3 for RGB images)
        C.output_classes = None  # Number of output classes (e.g., 10 for CIFAR-10)
        C.conv_layers = None     # List of tuples: (out_channels, kernel_size, stride, padding, pool_size)
        C.fc_layers = None       # List of fully connected layer sizes
        # Dropout hyperparameters
        C.dropout_p = 0.1        # Fully connected layer dropout
        return C

    def __init__(self, config):
        super().__init__()
        assert config.input_channels is not None
        assert config.output_classes is not None

        type_given = config.model_type is not None
        params_given = config.conv_layers is not None and config.fc_layers is not None
        assert type_given ^ params_given  # exactly one of these (XOR)
        
        if type_given:
            # Translate model_type into detailed layer configurations
            config.merge_from_dict({
                # Example CNN configurations
                'cnn-small': {
                    'conv_layers': [
                        (32, 3, 1, 1, 2),  # (out_channels, kernel_size, stride, padding, pool_size)
                        (64, 3, 1, 1, 2),
                    ],
                    'fc_layers': [128],  # Fully connected layers (hidden dimensions)
                },
                'cnn-medium': {
                    'conv_layers': [
                        (64, 3, 1, 1, 2),
                        (128, 3, 1, 1, 2),
                        (256, 3, 1, 1, 2),
                    ],
                    'fc_layers': [256, 128],
                },
                'cnn-large': {
                    'conv_layers': [
                        (128, 3, 1, 1, 2),
                        (256, 3, 1, 1, 2),
                        (512, 3, 1, 1, 2),
                    ],
                    'fc_layers': [512, 256, 128],
                },
            }[config.model_type])

        self.input_channels = config.input_channels
        self.output_classes = config.output_classes
        self.conv_layers = config.conv_layers
        self.fc_layers = config.fc_layers
        self.dropout_p = config.dropout_p

    @abstractmethod
    def forward(self, x):
        """Define the forward pass."""
        pass

# ----------------------------------------------------------------
# For child classes of BaseCNN
class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, ReLU and MaxPool2d layers"""

    def __init__(self, in_channels, out_channels, kernel_size, pool_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class FullyConnectedBlock(nn.Module):
    """Fully connected block with Linear and ReLU layers"""

    def __init__(self, in_features, out_features):
        super(FullyConnectedBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Flatten the input tensor if not already flat
        # x = x.view(-1, x.size(1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x