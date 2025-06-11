# ffnn.py
from collections import OrderedDict

import torch
import torch.nn as nn

from .base_ffnn import BaseFFNN


class SimpleFFNN(BaseFFNN):
    @staticmethod
    def get_default_config():
        C = BaseFFNN.get_default_config()
        # e.g. for MNIST: 1×28×28 → 784
        C.input_features = 1 * 28 * 28
        C.output_classes = 10
        C.fc_layers = [128, 64]  # two hidden layers
        C.dropout_p = 0.5
        return C

    def __init__(self, config):
        super().__init__(config)
        layers = OrderedDict()
        in_feats = config.input_features

        for i, h in enumerate(config.fc_layers, start=1):
            layers[f"fc{i}"] = nn.Linear(in_feats, h)
            layers[f"relu{i}"] = nn.ReLU()
            layers[f"dropout{i}"] = nn.Dropout(config.dropout_p)
            in_feats = h

        layers["out"] = nn.Linear(in_feats, config.output_classes)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
