import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from dd4ml.models.resnet.base_resnet import BaseResNet


class BigResNet(BaseResNet):
    def __init__(self, config, input_channels: int = 3, num_classes: int = 10):
        super().__init__(config)
        self.start = nn.Sequential(
            nn.Conv2d(self.config.input_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.stage2 = BasicBlock(64, 64)
        self.stage3 = BasicBlock(64, 64)
        self.stage4 = BasicBlock(
            64,
            128,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(64, 128, 1, 2, bias=False), nn.BatchNorm2d(128)
            ),
        )
        self.stage5 = BasicBlock(128, 128)
        self.stage6 = BasicBlock(
            128,
            256,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(128, 256, 1, 2, bias=False), nn.BatchNorm2d(256)
            ),
        )
        self.stage7 = BasicBlock(256, 256)
        self.finish = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256 * BasicBlock.expansion, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        return self.finish(x)

    def as_model_dict(self):
        md = {
            "start": {
                "callable": {
                    "object": nn.Sequential,
                    "settings": {
                        "modules": [
                            {
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
                            {
                                "object": nn.BatchNorm2d,
                                "settings": {"num_features": 64},
                            },
                            {"object": nn.ReLU, "settings": {"inplace": True}},
                            {
                                "object": nn.MaxPool2d,
                                "settings": {
                                    "kernel_size": 3,
                                    "stride": 2,
                                    "padding": 1,
                                },
                            },
                        ]
                    },
                },
                "dst": {"to": ["stage2"]},
                "rcv": {"src": [], "strategy": None},
                "stage": 0,
                "num_layer_shards": 1,
            },
            **{
                f"stage{i}": {
                    "callable": {"object": BasicBlock, "settings": settings},
                    "dst": {"to": [f"stage{i+1}" if i < 7 else "finish"]},
                    "rcv": {
                        "src": ["start"] if i == 2 else [f"stage{i-1}"],
                        "strategy": None,
                    },
                    "stage": i - 1,
                    "num_layer_shards": 1,
                }
                for i, settings in enumerate(
                    [
                        {
                            "inplanes": 64,
                            "planes": 64,
                            "stride": 1,
                            "downsample": None,
                        },  # i=2
                        {
                            "inplanes": 64,
                            "planes": 64,
                            "stride": 1,
                            "downsample": None,
                        },  # i=3
                        {
                            "inplanes": 64,
                            "planes": 128,
                            "stride": 2,
                            "downsample": nn.Sequential(
                                nn.Conv2d(64, 128, 1, 2, bias=False),
                                nn.BatchNorm2d(128),
                            ),
                        },  # i=4
                        {
                            "inplanes": 128,
                            "planes": 128,
                            "stride": 1,
                            "downsample": None,
                        },  # i=5
                        {
                            "inplanes": 128,
                            "planes": 256,
                            "stride": 2,
                            "downsample": nn.Sequential(
                                nn.Conv2d(128, 256, 1, 2, bias=False),
                                nn.BatchNorm2d(256),
                            ),
                        },  # i=6
                        {
                            "inplanes": 256,
                            "planes": 256,
                            "stride": 1,
                            "downsample": None,
                        },  # i=7
                    ],
                    start=2,
                )
            },
            "finish": {
                "callable": {
                    "object": nn.Sequential,
                    "settings": {
                        "modules": [
                            {
                                "object": nn.AdaptiveAvgPool2d,
                                "settings": {"output_size": (1, 1)},
                            },
                            {"object": nn.Flatten, "settings": {}},
                            {
                                "object": nn.Linear,
                                "settings": {
                                    "in_features": 256 * BasicBlock.expansion,
                                    "out_features": 10,
                                },
                            },
                        ]
                    },
                },
                "dst": {"to": []},
                "rcv": {"src": ["stage7"], "strategy": None},
                "stage": 7,
                "num_layer_shards": 1,
            },
        }
        self.model_dict = md
        self.set_stage()
        return md


if __name__ == "__main__":
    from dd4ml.utility import CfgNode

    # Instantiate your model (ensure `config` is defined appropriately)
    config = CfgNode()
    config.input_channels = 3  # Example input channels, adjust as needed
    config.output_classes = 10  # Example number of classes, adjust as needed
    config.model_type = None
    config.block_layers = 3
    config.num_filters = 64
    config.dropout_p = 0.1

    model = BigResNet(config)

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
