import matplotlib.pyplot as plt
import numpy as np
import math
from abc import  abstractmethod

class BCEnforceArchitecture(object):
    def __init__(self):
        pass

    @abstractmethod
    def bc_fun(self, coords):
        raise NotImplementedError

    @abstractmethod
    def lifting(self, coords):
        raise NotImplementedError

    @abstractmethod
    def apply(self, net_out, coords):
        raise NotImplementedError
        
        
class PDEBase(object):
    def __init__(self, bc_exact=None, bc_inexact=None, weights_data=1.0, weights_res=1.0):

        self.bc_exact        = bc_exact
        self.bc_inexact      = bc_inexact

        self.weights_data    = weights_data # maybe as a learnable parameter? Needs to be maximized.... 
        self.weights_res     = weights_res  # maybe as a learnable parameter? Needs to be maximized.... 

        self.has_analytical_sol   = False


    @abstractmethod
    def residual_loss(self, net_out, coords):
        raise NotImplementedError 

    @abstractmethod
    def residual(self, u, x):
        raise NotImplementedError 

    @abstractmethod
    def get_test_set(self):
        raise NotImplementedError 

    @abstractmethod
    def bc_loss(self, net):
        raise NotImplementedError   

    @abstractmethod
    def plot_results(self, domain, net, fig_name="Burgers.png", dpi=50):
        raise NotImplementedError  


    def computeRefError(self, net_out, vol):
        return torch.sqrt(torch.mean(torch.pow(net_out,2)) * vol)


    def pointwise_residual_eval(self, net, coords):
        net_out = net(coords)

        if(self.bc_exact is not None):
            # print("yes  here... ")
            net_out = self.bc_exact.apply_bc(net_out, coords)

        residual = self.residual(net_out, coords)

        return residual        


    def criterion(self, net, coords):

        net_out = net(coords)

        if(self.bc_exact is not None):
            # print("yes  here... ")
            net_out = self.bc_exact.apply_bc(net_out, coords)

        loss_residual   = self.weights_res * self.residual_loss(net_out, coords)
        # loss_residual   = net.weight_residual * self.residual_loss(net_out, coords)

        loss_data       = 0.0

        if(self.bc_inexact is not None):            
            loss_data = self.weights_data * self.bc_loss(net)
            # loss_data = net.weight_data * self.bc_loss(net)

        return net_out, loss_data + loss_residual, loss_residual





