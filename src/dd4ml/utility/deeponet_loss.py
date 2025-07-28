import torch.nn as nn
import torch.nn.functional as F

class DeepONetMSELoss(nn.Module):
    """Simple mean squared error loss for DeepONet outputs."""

    def forward(self, outputs, targets):
        return F.mse_loss(outputs, targets)
