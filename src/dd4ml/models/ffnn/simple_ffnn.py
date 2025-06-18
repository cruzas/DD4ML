# ffnn.py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_ffnn import BaseFFNN


class SimpleFFNN(BaseFFNN):
    @staticmethod
    def get_default_config():
        C = BaseFFNN.get_default_config()
        C.input_features = 1 * 28 * 28
        C.output_classes = 10
        C.fc_layers = [128, 64]
        C.dropout_p = 0.2
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
        x = self.net(x)
        return F.log_softmax(x, dim=1)
