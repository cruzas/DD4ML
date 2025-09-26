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
        C.fc_layers = [8, 8]
        C.dropout_p = 0.0
        return C

    def __init__(self, config):
        super().__init__(config)
        from ...utility.model_factory import create_fc_layers

        # Create the main layers using the factory function
        layers = create_fc_layers(
            input_features=config.input_features,
            fc_layers=config.fc_layers,
            dropout_p=config.dropout_p,
            use_sequential=False  # Get OrderedDict to add output layer
        )

        # Add output layer
        layers["out"] = nn.Linear(config.fc_layers[-1], config.output_classes)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return F.log_softmax(x, dim=1)

    def as_model_dict(self):
        pass
