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
        return self.net(x)
