import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import math
from PDEs.PDEBase import *
import pandas as pd
from pathlib import Path


class Poisson2D(PDEBase):

    def __init__(self, bc_exact=None, bc_inexact=None, weights_data=1.0, weights_res=1.0):
        super(Poisson2D, self).__init__(
            bc_exact, bc_inexact, weights_data, weights_res)
        self.has_analytical_sol = True

    def analytical_u(self, x):
        pi = torch.tensor(np.pi)
        return torch.sin(2*pi*x[:, 1])*(torch.tanh(10*x[:, 0]) + 0.1*torch.sin(2*pi*x[:, 0]))

    def residual(self, u, x):

        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(
            u), retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_x = grad_u[:, 0]
        grad_y = grad_u[:, 1]

        grad2_u_xx = torch.autograd.grad(grad_x, x, grad_outputs=torch.ones_like(
            u), retain_graph=True, create_graph=True, only_inputs=True)[0][:, 0]
        grad2_u_yy = torch.autograd.grad(grad_y, x, grad_outputs=torch.ones_like(
            u), retain_graph=True, create_graph=True, only_inputs=True)[0][:, 1]

        pi = torch.tensor(np.pi)

        f = 4*pi*pi*torch.sin(2*pi*x[:,1])*(torch.tanh(10*x[:,0]) + torch.sin(2*pi*x[:,0])/10) - torch.sin(2*pi*x[:,1]) * \
                (20*torch.tanh(10*x[:,0])*(10*torch.tanh(10*x[:,0])**2 - 10) - (2*pi*pi*torch.sin(2*pi*x[:,0]))/5)

        return (grad2_u_xx + grad2_u_yy + f)

    def residual_loss(self, net_out, coords):
        residual = self.residual(net_out, coords)
        return torch.mean((residual)**2)

    #def get_test_set(self):

    #    path = Path(__file__).parent / "exact_data/MinSurfExact.csv"
    #    df = pd.read_csv(path)

    #    x = torch.tensor(df['x'].values)
    #    y = torch.tensor(df['y'].values)
    #    sol = torch.tensor(df['sol'].values)

    #    num_samples = x.shape[0]

    #    x = torch.reshape(x, (num_samples, 1))
    #    y = torch.reshape(y, (num_samples, 1))
    #    sol = torch.reshape(sol, (num_samples, 1))

    #    test_set = torch.hstack((x, y, sol))

    #    return test_set

    def plot_results(self, domain,  net, fig_name="Poisson2D.png", dpi=100):

        X, Y = np.mgrid[domain.startPoint_x:domain.endPoint_x:100j,
                        domain.startPoint_y:domain.endPoint_y:100j]
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        positions = torch.tensor(positions)

        positions.requires_grad = True

        if torch.cuda.is_available():
            positions = positions.cuda()

        if (torch.cuda.is_available()):
            positions = positions.cuda()

        upred_domain = net(positions.double())
        if (self.bc_exact is not None):
            upred_domain = self.bc_exact.apply_bc(upred_domain, positions)

        pde_error = self.residual(upred_domain, positions)

        if torch.cuda.is_available():
            pde_error = pde_error.data.cpu().numpy()
        else:
            pde_error = pde_error.data.numpy()

        pde_error = pde_error.reshape(X.shape)

        # u_NN = net(positions.double())
        if torch.cuda.is_available():
            u_NN = upred_domain.data.cpu().numpy()
        else:
            u_NN = upred_domain.data.numpy()

        u_NN = u_NN.reshape(X.shape)

        fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
        ax = axs[0]

        pcm = ax.plot_surface(X, Y, u_NN)
        ax.set_title('Approx. solution', fontsize=10)

        ax = axs[1]
        pcm = ax.plot_surface(X, Y, pde_error)
        ax.set_title('Residual', fontsize=10)

        fig.tight_layout(pad=5)
        plt.savefig(fig_name)


class BC_Poisson2D(BCEnforceArchitecture):
    def __init__(self, start_point, end_point):
        super(BC_Poisson2D, self).__init__()

        self.startPoint_x = torch.tensor(start_point[0])
        self.startPoint_y = torch.tensor(start_point[1])

        #  end point is upper right point of the bounding box
        self.endPoint_x = torch.tensor(end_point[0])
        self.endPoint_y = torch.tensor(end_point[1])

    def bc_fun(self, coords):
        pi = torch.tensor(np.pi)
        return torch.sin(2*pi*coords[:, 1])*(torch.tanh(10*coords[:, 0]) + 0.1*torch.sin(2*pi*coords[:, 0]))

    def lifting(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]

        return ((x - self.startPoint_x) * (self.endPoint_x - x) * (y - self.startPoint_y) * (self.endPoint_y - y))

    def apply_bc(self, net_out, coords):
        out = self.bc_fun(coords) + (net_out[:, 0]*self.lifting(coords))
        return out


if __name__ == '__main__':

    pde = Poisson2D()

    # x = np.linspace(0, 1, 100, endpoint=True)
    # x = x.reshape((-1, 1))
    # x = torch.tensor(x)

    # exact = pde.analytical_sol(x)
    # plt.plot(x.detach().numpy(), exact.detach().numpy())
    # plt.ylabel('Sol')
    # plt.show()

    train_samples = np.array([10, 10])
    test_samples = np.array([50, 50])
    start_point = np.array([-1., -1.])
    end_point = np.array([1., 1.])

    dim = 2
    penaltyPara = 1e2
