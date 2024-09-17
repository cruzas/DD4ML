import torch
import torch.nn as nn
import math
from torchsummary import summary
import torch.nn.init as init
import torch.distributed as dist
from models.ResNetDenseConstantWidth import *


class PODDeepOnet(nn.Module):

    def __init__(self, POD_basis, POD_mean, branch_net):
        super(PODDeepOnet, self).__init__()

        self.POD_basis = nn.Parameter(POD_basis)            
        self.POD_basis.requires_grad = False
        
        self.POD_mean = nn.Parameter(POD_mean)
        self.POD_mean.requires_grad = False 

        self.num_basis = self.POD_basis.shape[1]

        self.branch_net = branch_net
        self.branch_net.init_params()

    def forward(self, x):

        branch_outputs_net = self.branch_net(x[0])  

        u_pred  = branch_outputs_net @ self.POD_basis.T 
        u_pred  = u_pred/self.num_basis + self.POD_mean

        return u_pred

    def get_avg(self):
        return self.branch_net.get_avg()

    def get_num_layers(self):
        return self.branch_net.get_num_layers()

    def extract_trainable_params(self, sbd_id, num_subdomains, overlap_width=0):
        return self.branch_net.extract_trainable_params(sbd_id, num_subdomains, overlap_width)

    def extract_coarse_trainable_params(self, num_subdomains, overlap_width=0):
        return self.branch_net.extract_coarse_trainable_params(num_subdomains, overlap_width=0)

    def extract_multilevel_trainable_params(self, num_subdomains, level=1):
        return self.branch_net.extract_multilevel_trainable_params(num_subdomains, level)

    def extract_sbd_quantity(self, sbd_id, num_subdomains, all_tensors, overlap_width=0):
        return self.branch_net.extract_sbd_quantity(sbd_id, num_subdomains, all_tensors, overlap_width=0)

    def print_decomposition(self, num_subdomains): 
        return self.branch_net.print_decomposition(num_subdomains)



if __name__ == '__main__':
    print("Not implemented... ")
    exit(0)



