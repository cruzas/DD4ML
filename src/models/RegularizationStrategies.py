import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init


# TODO, verify GPU sum 
class L2Regularization():
    def __init__(self, reg) -> None:
        self.reg = reg

    def __call__(self, net):
        l2_norm= sum(p.pow(2.0).sum() for p in net.parameters() if p.requires_grad)

        return self.reg*l2_norm


