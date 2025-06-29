import torch
import torch.nn as nn

from .base_ffnn import BaseFFNN

class PINNFFNN(BaseFFNN):
    """Small fully-connected network for PINN regression tasks."""

    @staticmethod
    def get_default_config():
        C = BaseFFNN.get_default_config()
        C.input_features = 1
        C.output_classes = 1
        C.fc_layers = [20, 20]
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
