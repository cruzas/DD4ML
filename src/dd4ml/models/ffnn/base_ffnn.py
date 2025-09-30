# base_ffnn.py
from abc import abstractmethod

import torch.nn as nn

from dd4ml.models.base_model import BaseModel


class BaseFFNN(BaseModel):
    """Abstract Base Class for feedforward (fully-connected) models."""

    @staticmethod
    def get_default_config():
        C = BaseModel.get_default_config()
        C.model_type = None
        C.input_features = None  # dimension of flattened input
        C.output_classes = None
        C.fc_layers = None  # list of hidden sizes
        C.width = None  # width of hidden layers (overrides fc_layers values)
        C.dropout_p = 0.1
        return C

    def __init__(self, config):
        super().__init__(config)
        assert config.input_features is not None
        assert config.output_classes is not None

        type_given = config.model_type is not None
        params_given = config.fc_layers is not None
        assert type_given ^ params_given  # exactly one

        if type_given:
            config.merge_from_dict(
                {
                    "ffnn-small": {"fc_layers": [128]},
                    "ffnn-medium": {"fc_layers": [256, 128]},
                    "ffnn-large": {"fc_layers": [512, 256, 128]},
                }[config.model_type]
            )

        # If width is specified, override fc_layers with uniform width
        if config.width is not None and config.fc_layers is not None:
            num_layers = len(config.fc_layers)
            config.fc_layers = [config.width] * num_layers

        self.input_features = config.input_features
        self.output_classes = config.output_classes
        self.fc_layers = config.fc_layers
        self.dropout_p = config.dropout_p

    @abstractmethod
    def forward(self, x):
        pass
