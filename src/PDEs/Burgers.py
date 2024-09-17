import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PDEs.PDEBase import *
import pandas as pd
from pathlib import Path


class Burgers(PDEBase):
    def __init__(self, nu=0.01, bc_exact=None, bc_inexact=None, weights_data=1.0, weights_res=1.0):
        super(Burgers, self).__init__(bc_exact, bc_inexact, weights_data, weights_res)

        self.nu = torch.tensor(nu)
        self.pi = torch.tensor(np.pi)


    def residual(self, u, x):

        grad_u      = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_t      = grad_u[:, 0]
        grad_x      = grad_u[:, 1]

        gradu_xx    = torch.autograd.grad(grad_x, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True, only_inputs=True)[0][:, 1]     

        return grad_t + (u*grad_x) - ((self.nu/self.pi)* gradu_xx)

    
    def residual_loss(self, net_out, coords):
        residual = self.residual(net_out, coords)
        return torch.mean((residual)**2)


    def get_test_set(self):

        path = Path(__file__).parent / "exact_data/BurgersExact.csv"
        df = pd.read_csv(path)

        x = torch.tensor(df['t'].values)
        y = torch.tensor(df['x'].values)
        sol = torch.tensor(df['sol'].values)

        num_samples = x.shape[0]

        x   = torch.reshape(x, (num_samples, 1))
        y   = torch.reshape(y, (num_samples, 1))
        sol = torch.reshape(sol, (num_samples, 1))

        test_set = torch.hstack((x, y, sol))

        return test_set        


    def plot_results(self, domain,  net, fig_name="Burgers.png", dpi=50):
        fig, axs = plt.subplots(nrows=1, figsize=(3, 3), dpi=dpi)

        x = domain.x_test.double()
        x.requires_grad = True

        if(torch.cuda.is_available()):
            x = x.cuda()                  

        u_nn_test = net(x)    
        if type(u_nn_test) is list and len(u_nn_test) == 1:
            u_nn_test = u_nn_test[0]
        if(self.bc_exact is not None):
            u_nn_test = self.bc_exact.apply_bc(u_nn_test, x)

        u_NN = u_nn_test.reshape(x[:,1].shape)
        
        
        if(torch.cuda.is_available()):
            u_NN = u_NN.cpu().detach().numpy()
        else:
            u_NN = u_NN.detach().numpy()


        u_NN = u_NN.reshape((domain.num_test_samples_shapes[1], domain.num_test_samples_shapes[0]))
        fig = plt.figure()
        gs0 = gridspec.GridSpec(1,1)
        gs0.update(top=0.95, bottom=0.45, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :], label='u')

        h = ax.imshow(u_NN, interpolation='nearest', cmap='rainbow', 
                      extent=[0, 1, -1, 1], 
                      origin='lower', 
                      aspect='auto', label='u')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)

        ax.set_title('PINN solution', fontsize=10)
        plt.savefig(fig_name)




class BC_Burgers(BCEnforceArchitecture):
    def __init__(self, start_point, end_point):
        super(BC_Burgers, self).__init__()

        self.startPoint_x = torch.tensor(start_point[0])  # xmin
        self.startPoint_y = torch.tensor(start_point[1])  # ymin

        # end point is upper right point of the bounding box
        self.endPoint_x = torch.tensor(end_point[0])    # xmax
        self.endPoint_y = torch.tensor(end_point[1])    # xmin
        self.pi         = torch.tensor(np.pi)
  

    def bc_fun(self, coords):
        return torch.sin(-self.pi*coords[:, 1])


    def lifting(self, coords):
        x = coords[:,0]
        y = coords[:,1]
        return (y - self.startPoint_y) * (self.endPoint_y - y) * (x - self.startPoint_x)


    # simple trick due to tailing 
    def apply_bc(self, net_out, coords):
        out = self.bc_fun(coords) + (net_out[:,0]*self.lifting(coords))
        return out



if __name__ == '__main__':


    pde = Burgers()
    
    train_samples = np.array([10, 10])
    test_samples = np.array([50, 50])
    start_point = np.array([0., 0.])
    end_point = np.array([1., 1.])

    dim = 2
    penaltyPara = 1e2
