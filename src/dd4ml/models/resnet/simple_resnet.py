import torch.nn as nn
import torch.nn.functional as F

from dd4ml.models.resnet.base_resnet import BaseResNet
from torchvision.models import resnet18


# Basic residual block.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.start = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.start(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# Helper module that encapsulates a ResNet layer.
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, block, out_channels, blocks, stride):
        super(ResNetLayer, self).__init__()
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(out_channels * block.expansion, out_channels))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


# Simple flatten module.
class FlattenBlock(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# ResNet model (ResNet-18 variant).
class SimpleResNet(BaseResNet):
    @staticmethod
    def get_default_config():
        C = BaseResNet.get_default_config()
        return C

    def __init__(self, config, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10):
        super().__init__(config)
        # For all others
        # self.backbone = resnet18(pretrained=False)
        
        # For APTS_IP
        self.layers_config = layers  # store configuration for use in as_model_dict
        self.in_channels = 64
        self.start = nn.Conv2d(
            config.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResNetLayer(self.in_channels, block, 64, layers[0], stride=1)
        self.in_channels = 64 * block.expansion
        self.layer2 = ResNetLayer(self.in_channels, block, 128, layers[1], stride=2)
        self.in_channels = 128 * block.expansion
        self.layer3 = ResNetLayer(self.in_channels, block, 256, layers[2], stride=2)
        self.in_channels = 256 * block.expansion
        self.layer4 = ResNetLayer(self.in_channels, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = FlattenBlock()
        self.finish = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        # For APTS_IP
        # return self.backbone(x)
        
        # For all others
        x = self.relu(self.bn1(self.start(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.finish(x)

    def as_model_dict(self):
        model_dict = {
            "start": {
                "callable": {
                    "object": nn.Conv2d,
                    "settings": {
                        "in_channels": self.config.input_channels,
                        "out_channels": 64,
                        "kernel_size": 7,
                        "stride": 2,
                        "padding": 3,
                        "bias": False,
                    },
                },
                "dst": {"to": ["bn1"]},
                "rcv": {"src": [], "strategy": None},
                "stage": 0,
                "num_layer_shards": 1,
            },
            "bn1": {
                "callable": {
                    "object": nn.BatchNorm2d,
                    "settings": {"num_features": 64},
                },
                "dst": {"to": ["relu"]},
                "rcv": {"src": ["start"], "strategy": None},
                "stage": 1,
                "num_layer_shards": 1,
            },
            "relu": {
                "callable": {
                    "object": nn.ReLU,
                    "settings": {"inplace": True},
                },
                "dst": {"to": ["maxpool"]},
                "rcv": {"src": ["bn1"], "strategy": None},
                "stage": 2,
                "num_layer_shards": 1,
            },
            "maxpool": {
                "callable": {
                    "object": nn.MaxPool2d,
                    "settings": {"kernel_size": 3, "stride": 2, "padding": 1},
                },
                "dst": {"to": ["layer1"]},
                "rcv": {"src": ["relu"], "strategy": None},
                "stage": 3,
                "num_layer_shards": 1,
            },
            "layer1": {
                "callable": {
                    "object": ResNetLayer,
                    "settings": {
                        "in_channels": 64,
                        "block": BasicBlock,
                        "out_channels": 64,
                        "blocks": self.layers_config[0],
                        "stride": 1,
                    },
                },
                "dst": {"to": ["layer2"]},
                "rcv": {"src": ["maxpool"], "strategy": None},
                "stage": 4,
                "num_layer_shards": 1,
            },
            "layer2": {
                "callable": {
                    "object": ResNetLayer,
                    "settings": {
                        "in_channels": 64 * BasicBlock.expansion,
                        "block": BasicBlock,
                        "out_channels": 128,
                        "blocks": self.layers_config[1],
                        "stride": 2,
                    },
                },
                "dst": {"to": ["layer3"]},
                "rcv": {"src": ["layer1"], "strategy": None},
                "stage": 5,
                "num_layer_shards": 1,
            },
            "layer3": {
                "callable": {
                    "object": ResNetLayer,
                    "settings": {
                        "in_channels": 128 * BasicBlock.expansion,
                        "block": BasicBlock,
                        "out_channels": 256,
                        "blocks": self.layers_config[2],
                        "stride": 2,
                    },
                },
                "dst": {"to": ["layer4"]},
                "rcv": {"src": ["layer2"], "strategy": None},
                "stage": 6,
                "num_layer_shards": 1,
            },
            "layer4": {
                "callable": {
                    "object": ResNetLayer,
                    "settings": {
                        "in_channels": 256 * BasicBlock.expansion,
                        "block": BasicBlock,
                        "out_channels": 512,
                        "blocks": self.layers_config[3],
                        "stride": 2,
                    },
                },
                "dst": {"to": ["avgpool"]},
                "rcv": {"src": ["layer3"], "strategy": None},
                "stage": 7,
                "num_layer_shards": 1,
            },
            "avgpool": {
                "callable": {
                    "object": nn.AdaptiveAvgPool2d,
                    "settings": {"output_size": (1, 1)},
                },
                "dst": {"to": ["flatten"]},
                "rcv": {"src": ["layer4"], "strategy": None},
                "stage": 8,
                "num_layer_shards": 1,
            },
            "flatten": {
                "callable": {
                    "object": FlattenBlock,
                    "settings": {},
                },
                "dst": {"to": ["finish"]},
                "rcv": {"src": ["avgpool"], "strategy": None},
                "stage": 9,
                "num_layer_shards": 1,
            },
            "finish": {
                "callable": {
                    "object": nn.Linear,
                    "settings": {
                        "in_features": 512 * BasicBlock.expansion,
                        "out_features": 10,
                    },
                },
                "dst": {"to": []},
                "rcv": {"src": ["flatten"], "strategy": None},
                "stage": 10,
                "num_layer_shards": 1,
            },
        }
        self.model_dict = model_dict
        self.set_stage()
        return self.model_dict
