import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_cnn import BaseCNN


# Define callable classes
class ConvBNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, pool_size, stride
    ):
        super(ConvBNBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class FlattenBlock(nn.Module):
    def __init__(self):
        super(FlattenBlock, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, activation="relu"):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU() if activation == "relu" else None

    def forward(self, x):
        x = self.fc(x)
        if self.activation:
            x = self.activation(x)
        return x


class DropoutBlock(nn.Module):
    def __init__(self, p):
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)


class IdentityBlock(nn.Module):
    def __init__(self):
        super(IdentityBlock, self).__init__()

    def forward(self, x):
        return x


class SimpleCNN(BaseCNN):

    @staticmethod
    def get_default_config():
        C = BaseCNN.get_default_config()
        # Default to MNIST settings; update these for CIFAR10 as needed.
        C.input_channels = 1
        C.input_height = 28
        C.input_width = 28
        return C

    def __init__(self, config):
        super().__init__(config)

        self.conv1 = nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically determine flattened feature size
        dummy = torch.zeros(
            1, config.input_channels, config.input_height, config.input_width
        )
        dummy = self.pool(F.relu(self.batchnorm1(self.conv1(dummy))))
        dummy = self.pool(F.relu(self.batchnorm2(self.conv2(dummy))))
        dummy = self.pool(F.relu(self.batchnorm3(self.conv3(dummy))))
        self.flattened_size = dummy.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def as_model_dict(self):
        model_dict = {
            "start": {
                "callable": {
                    "object": ConvBNBlock,
                    "settings": {
                        "in_channels": self.config.input_channels,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "padding": 1,
                        "pool_size": 2,
                        "stride": 1,
                    },
                },
                "dst": {"to": ["conv2"]},
                "rcv": {"src": [], "strategy": None},
                "stage": 0,
                "num_layer_shards": 1,
            },
            "conv2": {
                "callable": {
                    "object": ConvBNBlock,
                    "settings": {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "padding": 1,
                        "pool_size": 2,
                        "stride": 1,
                    },
                },
                "dst": {"to": ["conv3"]},
                "rcv": {"src": ["start"], "strategy": None},
                "stage": 1,
                "num_layer_shards": 1,
            },
            "conv3": {
                "callable": {
                    "object": ConvBNBlock,
                    "settings": {
                        "in_channels": 64,
                        "out_channels": 128,
                        "kernel_size": 3,
                        "padding": 1,
                        "pool_size": 2,
                        "stride": 1,
                    },
                },
                "dst": {"to": ["flatten"]},
                "rcv": {"src": ["conv2"], "strategy": None},
                "stage": 2,
                "num_layer_shards": 1,
            },
            "flatten": {
                "callable": {
                    "object": FlattenBlock,
                    "settings": {},
                },
                "dst": {"to": ["fc1"]},
                "rcv": {"src": ["conv3"], "strategy": None},
                "stage": 3,
                "num_layer_shards": 1,
            },
            "fc1": {
                "callable": {
                    "object": FCBlock,
                    "settings": {
                        "in_features": self.flattened_size,
                        "out_features": 256,
                        "activation": "relu",
                    },
                },
                "dst": {"to": ["dropout"]},
                "rcv": {"src": ["flatten"], "strategy": None},
                "stage": 4,
                "num_layer_shards": 1,
            },
            "dropout": {
                "callable": {
                    "object": DropoutBlock,
                    "settings": {
                        "p": 0.5,
                    },
                },
                "dst": {"to": ["fc2"]},
                "rcv": {"src": ["fc1"], "strategy": None},
                "stage": 5,
                "num_layer_shards": 1,
            },
            "fc2": {
                "callable": {
                    "object": FCBlock,
                    "settings": {
                        "in_features": 256,
                        "out_features": 10,
                        "activation": None,
                    },
                },
                "dst": {"to": ["finish"]},
                "rcv": {"src": ["dropout"], "strategy": None},
                "stage": 6,
                "num_layer_shards": 1,
            },
            "finish": {
                "callable": {
                    "object": IdentityBlock,
                    "settings": {},
                },
                "dst": {"to": []},
                "rcv": {"src": ["fc2"], "strategy": None},
                "stage": 7,
                "num_layer_shards": 1,
            },
        }

        self.model_dict = model_dict
        self.set_stage()
        return self.model_dict
