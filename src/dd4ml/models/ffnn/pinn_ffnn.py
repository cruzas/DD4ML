import torch
import torch.nn as nn

from .base_ffnn import BaseFFNN


class PINNFFNN(BaseFFNN):
    """Fully-connected network with at least eight hidden layers for PINN regression."""

    @staticmethod
    def get_default_config():
        C = BaseFFNN.get_default_config()
        C.input_features = 1
        C.output_classes = 1
        # Eight hidden layers of width 20
        C.fc_layers = [20] * 8
        C.dropout_p = 0.0
        return C

    def __init__(self, config):
        super().__init__(config)
        layers = []
        in_feats = self.input_features
        for h in self.fc_layers:
            layers.append(nn.Linear(in_feats, h))
            layers.append(nn.Tanh())
            in_feats = h
        layers.append(nn.Linear(in_feats, self.output_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def as_model_dict(self):
        cfg = self.config
        md = {
            "start": {
                "callable": {
                    "object": nn.Sequential,
                    "settings": {
                        "modules": [
                            {
                                "object": nn.Linear,
                                "settings": {
                                    "in_features": cfg.input_features,
                                    "out_features": cfg.fc_layers[0],
                                },
                            },
                            {"object": nn.Tanh, "settings": {}},
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
                    "callable": {
                        "object": nn.Sequential,
                        "settings": {
                            "modules": [
                                {
                                    "object": nn.Linear,
                                    "settings": {
                                        "in_features": cfg.fc_layers[i - 2],
                                        "out_features": cfg.fc_layers[i - 1],
                                    },
                                },
                                {"object": nn.Tanh, "settings": {}},
                            ]
                        },
                    },
                    "dst": {
                        "to": [f"stage{i+1}"] if i < len(cfg.fc_layers) else ["finish"]
                    },
                    "rcv": {
                        "src": ["start"] if i == 2 else [f"stage{i-1}"],
                        "strategy": None,
                    },
                    "stage": i - 1,
                    "num_layer_shards": 1,
                }
                for i in range(2, len(cfg.fc_layers) + 1)
            },
            "finish": {
                "callable": {
                    "object": nn.Linear,
                    "settings": {
                        "in_features": cfg.fc_layers[-1],
                        "out_features": cfg.output_classes,
                    },
                },
                "dst": {"to": []},
                "rcv": {"src": [f"stage{len(cfg.fc_layers)}"], "strategy": None},
                "stage": len(cfg.fc_layers),
                "num_layer_shards": 1,
            },
        }

        self.model_dict = md
        self.set_stage()
        return md
