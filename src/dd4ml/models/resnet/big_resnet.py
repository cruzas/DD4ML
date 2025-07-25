import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from dd4ml.models.resnet.base_resnet import BaseResNet

class BigResNet(BaseResNet):
    def __init__(self, config, input_channels: int = 3, num_classes: int = 10):
        super().__init__(config)
        # Stage 1: conv → BN → ReLU
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages 2–7: six BasicBlock modules
        self.stage2 = BasicBlock(64, 64)
        self.stage3 = BasicBlock(64, 64)
        self.stage4 = BasicBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        ))
        self.stage5 = BasicBlock(128, 128)
        self.stage6 = BasicBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        ))
        self.stage7 = BasicBlock(256, 256)

        # Stage 8: global pool → FC
        self.stage8 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        return self.stage8(x)

    def as_model_dict(self):
        md = {
            "start": {
                "callable": {
                    "object": nn.Conv2d,
                    "settings": {
                        "in_channels":   self.config.input_channels,
                        "out_channels":  64,
                        "kernel_size":   7,
                        "stride":        2,
                        "padding":       3,
                        "bias":          False
                    }
                },
                "dst":      {"to": ["bn1"]},
                "rcv":      {"src": [],           "strategy": None},
                "stage":    0,
                "num_layer_shards": 1
            },
            "bn1": {
                "callable": {
                    "object": nn.BatchNorm2d,
                    "settings": {"num_features": 64}
                },
                "dst":      {"to": ["relu"]},
                "rcv":      {"src": ["start"],    "strategy": None},
                "stage":    1,
                "num_layer_shards": 1
            },
            "relu": {
                "callable": {
                    "object": nn.ReLU,
                    "settings": {"inplace": True}
                },
                "dst":      {"to": ["maxpool"]},
                "rcv":      {"src": ["bn1"],      "strategy": None},
                "stage":    2,
                "num_layer_shards": 1
            },
            "maxpool": {
                "callable": {
                    "object": nn.MaxPool2d,
                    "settings": {"kernel_size": 3, "stride": 2, "padding": 1}
                },
                "dst":      {"to": ["stage2"]},
                "rcv":      {"src": ["relu"],     "strategy": None},
                "stage":    3,
                "num_layer_shards": 1
            },
            # each BasicBlock as its own stage
            "stage2": {
                "callable": {
                    "object": BasicBlock,
                    "settings": {
                        "inplanes": 64,
                        "planes":   64,
                        "stride":   1,
                        "downsample": None
                    }
                },
                "dst":      {"to": ["stage3"]},
                "rcv":      {"src": ["maxpool"],  "strategy": None},
                "stage":    4,
                "num_layer_shards": 1
            },
            "stage3": {
                "callable": {
                    "object": BasicBlock,
                    "settings": {
                        "inplanes": 64,
                        "planes":   64,
                        "stride":   1,
                        "downsample": None
                    }
                },
                "dst":      {"to": ["stage4"]},
                "rcv":      {"src": ["stage2"],   "strategy": None},
                "stage":    5,
                "num_layer_shards": 1
            },
            "stage4": {
                "callable": {
                    "object": BasicBlock,
                    "settings": {
                        "inplanes": 64,
                        "planes":   128,
                        "stride":   2,
                        "downsample": nn.Sequential(
                            nn.Conv2d(64, 128, 1, stride=2, bias=False),
                            nn.BatchNorm2d(128)
                        )
                    }
                },
                "dst":      {"to": ["stage5"]},
                "rcv":      {"src": ["stage3"],   "strategy": None},
                "stage":    6,
                "num_layer_shards": 1
            },
            "stage5": {
                "callable": {
                    "object": BasicBlock,
                    "settings": {
                        "inplanes": 128,
                        "planes":   128,
                        "stride":   1,
                        "downsample": None
                    }
                },
                "dst":      {"to": ["stage6"]},
                "rcv":      {"src": ["stage4"],   "strategy": None},
                "stage":    7,
                "num_layer_shards": 1
            },
            "stage6": {
                "callable": {
                    "object": BasicBlock,
                    "settings": {
                        "inplanes": 128,
                        "planes":   256,
                        "stride":   2,
                        "downsample": nn.Sequential(
                            nn.Conv2d(128, 256, 1, stride=2, bias=False),
                            nn.BatchNorm2d(256)
                        )
                    }
                },
                "dst":      {"to": ["stage7"]},
                "rcv":      {"src": ["stage5"],   "strategy": None},
                "stage":    8,
                "num_layer_shards": 1
            },
            "stage7": {
                "callable": {
                    "object": BasicBlock,
                    "settings": {
                        "inplanes": 256,
                        "planes":   256,
                        "stride":   1,
                        "downsample": None
                    }
                },
                "dst":      {"to": ["avgpool"]},
                "rcv":      {"src": ["stage6"],   "strategy": None},
                "stage":    9,
                "num_layer_shards": 1
            },
            "avgpool": {
                "callable": {
                    "object": nn.AdaptiveAvgPool2d,
                    "settings": {"output_size": (1, 1)}
                },
                "dst":      {"to": ["flatten"]},
                "rcv":      {"src": ["stage7"],   "strategy": None},
                "stage":    10,
                "num_layer_shards": 1
            },
            "flatten": {
                "callable": {
                    "object": nn.Flatten,
                    "settings": {}
                },
                "dst":      {"to": ["finish"]},
                "rcv":      {"src": ["avgpool"],  "strategy": None},
                "stage":    11,
                "num_layer_shards": 1
            },
            "finish": {
                "callable": {
                    "object": nn.Linear,
                    "settings": {
                        "in_features": 256 * BasicBlock.expansion,
                        "out_features": 10
                    }
                },
                "dst":      {"to": []},
                "rcv":      {"src": ["flatten"],  "strategy": None},
                "stage":    12,
                "num_layer_shards": 1
            },
        }
        self.model_dict = md
        self.set_stage()
        return md
