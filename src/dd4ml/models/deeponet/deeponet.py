import torch
import torch.nn as nn
from dd4ml.models.base_model import BaseModel

class DeepONet(BaseModel):
    """Minimal DeepONet implementation with branch and trunk networks."""

    @staticmethod
    def get_default_config():
        C = BaseModel.get_default_config()
        C.branch_hidden = [64, 64]
        C.trunk_hidden = [64, 64]
        C.branch_input_dim = None
        C.trunk_input_dim = 1
        C.output_dim = 1
        return C

    def __init__(self, config):
        super().__init__(config)
        assert config.branch_input_dim is not None, "branch_input_dim must be set"

        self.branch_net = self._make_mlp(config.branch_input_dim, config.branch_hidden)
        self.trunk_net = self._make_mlp(config.trunk_input_dim, config.trunk_hidden)
        p = config.branch_hidden[-1]
        assert config.trunk_hidden[-1] == p, "branch and trunk output dims must match"
        self.p = p

    def _make_mlp(self, in_dim, hidden):
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        return nn.Sequential(*layers)

    def forward(self, inputs):
        branch, trunk = inputs
        b = self.branch_net(branch)
        t = self.trunk_net(trunk)
        y = (b * t).sum(dim=1, keepdim=True)
        return y
