import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ffnn import BaseFFNN

class XLFFNN(BaseFFNN):
    @staticmethod
    def get_default_config():
        C = BaseFFNN.get_default_config()
        C.input_features  = 1 * 28 * 28
        C.output_classes  = 10
        C.fc_layers       = [128] * 32
        C.dropout_p       = 0.0
        return C

    def __init__(self, config):
        super().__init__(config)
        # build 32 hidden stages
        in_feats = config.input_features
        for i, h in enumerate(config.fc_layers, start=1):
            setattr(self, f"stage{i}", nn.Sequential(
                nn.Linear(in_feats, h),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_p)
            ))
            in_feats = h
        # final classifier
        self.finish = nn.Linear(in_feats, config.output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i in range(1, len(self.config.fc_layers) + 1):
            x = getattr(self, f"stage{i}")(x)
        return F.log_softmax(self.finish(x), dim=1)

    def _create_fc_stages(self):
        """Create FC stages using the factory helper."""
        from ...utility.model_factory import create_fc_stage_modules
        return create_fc_stage_modules(self.config)

    def as_model_dict(self):
        cfg = self.config
        md = {
            "start": {
                "callable": {
                    "object": nn.Sequential,
                    "settings": {
                        "modules": [
                            {
                                "object": nn.Flatten,
                                "settings": {}
                            },
                            {
                                "object": nn.Linear,
                                "settings": {
                                    "in_features":  cfg.input_features,
                                    "out_features": cfg.fc_layers[0]
                                }
                            },
                            {
                                "object": nn.ReLU,
                                "settings": {"inplace": True}
                            },
                            {
                                "object": nn.Dropout,
                                "settings": {"p": cfg.dropout_p}
                            }
                        ]
                    }
                },
                "dst":      {"to": ["stage2"]},
                "rcv":      {"src": [],           "strategy": None},
                "stage":    0,
                "num_layer_shards": 1
            },
            **self._create_fc_stages(),
            "finish": {
                "callable": {
                    "object": nn.Linear,
                    "settings": {
                        "in_features":  cfg.fc_layers[-1],
                        "out_features": cfg.output_classes
                    }
                },
                "dst":      {"to": []},
                "rcv":      {"src": [f"stage{len(cfg.fc_layers)}"], "strategy": None},
                "stage":    len(cfg.fc_layers),
                "num_layer_shards": 1
            }
        }

        self.model_dict = md
        self.set_stage()
        return md