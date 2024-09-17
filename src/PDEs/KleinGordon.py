import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import math
from datasets.Domain2D import *
from PDEs.PDEBase import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# https://www.sciencedirect.com/science/article/pii/S0307904X14006064
# Example 1 (Section 3.1)


class BC_Data_KleinGordon1D_plus_time(object):
    def __init__(self, t_zero, xm1, xp1):

        self.inputs = np.concatenate((t_zero, xm1), axis=0)
        self.inputs = torch.tensor(np.concatenate((self.inputs, xp1), axis=0))
        self.inputs.requires_grad = True

        if(torch.cuda.is_available()):
            self.inputs = self.inputs.cuda()

        t_zero_target = t_zero[:, 1].clone().detach()

        xm1_target = -torch.cos(xm1[:, 0])
        xp1_target = torch.cos(xm1[:, 0])

        self.targets = np.concatenate((t_zero_target, xm1_target), axis=0)
        self.targets = np.concatenate((self.targets, xp1_target), axis=0)
        self.targets = torch.tensor(self.targets.reshape(len(self.targets), 1))

        if(torch.cuda.is_available()):
            self.targets = self.targets.cuda()

        self.inputs_time_deriv = t_zero.clone().detach()
        self.inputs_time_deriv.requires_grad = True

        if(torch.cuda.is_available()):
            self.inputs_time_deriv = self.inputs_time_deriv.cuda()

        self.targets_time_deriv = 0.0*t_zero[:, 1]
        self.targets_time_deriv = self.targets_time_deriv.clone().detach()

        if(torch.cuda.is_available()):
            self.targets_time_deriv = self.targets_time_deriv.cuda()


class KleinGordon(PDEBase):
    def __init__(self, alpha=-1.0, beta=0.0, gamma=1.0, k=2, bc_exact=None, bc_inexact=None, weights_data=1.0, weights_res=1.0):
        super(KleinGordon, self).__init__(
            bc_exact, bc_inexact, weights_data, weights_res)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.has_analytical_sol = True

    # analytical solution -------------------------------------
    def analytical_u(self, x):
        return x[:,1] * torch.cos(x[:, 0])


    def residual(self, u, x):
        # just to have dimension we need
        # u = u[:, 0]

        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(
            u), retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_t = grad_u[:, 0]
        grad_x = grad_u[:, 1]

        grad2_u_tt = torch.autograd.grad(grad_t, x, grad_outputs=torch.ones_like(
            u), retain_graph=True, create_graph=True, only_inputs=True)[0][:, 0]
        grad2_u_xx = torch.autograd.grad(grad_x, x, grad_outputs=torch.ones_like(
            u), retain_graph=True, create_graph=True, only_inputs=True)[0][:, 1]

        rhs = (-x[:, 1] * torch.cos(x[:, 0])) + \
            (x[:, 1]**2 * torch.pow(torch.cos(x[:, 0]), 2))

        return grad2_u_tt + (self.alpha*grad2_u_xx) + (self.beta * u) + (self.gamma*torch.pow(u, self.k)) - rhs

    def time_deriv(self, u, x):
        # just to have dimension we need
        u = u[:, 0]

        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(
            u), retain_graph=True, create_graph=True, only_inputs=True)[0]
        grad_t = grad_u[:, 0]

        return grad_t

    def residual_loss(self, net_out, coords):
        residual = self.residual(net_out, coords)
        return torch.mean((residual)**2)

    def bc_loss(self, net):
        loss_mse = nn.MSELoss()

        u_BC = net(self.bc_inexact.inputs)
        u_t = self.time_deriv(
            net(self.bc_inexact.inputs_time_deriv), self.bc_inexact.inputs_time_deriv)

        return loss_mse(u_BC, self.bc_inexact.targets) + loss_mse(u_t, self.bc_inexact.targets_time_deriv)

    def plot_results(self, domain,  net, fig_name="KleinGordon", dpi=100):

        fig, axs = plt.subplots(nrows=1, figsize=(3, 3), dpi=dpi)

        x = domain.x_test.double()
        x.requires_grad = True

        if(torch.cuda.is_available()):
            x = x.cuda()

        u_nn_test = net(x)
        if(self.bc_exact is not None):
            u_nn_test = self.bc_exact.apply_bc(u_nn_test, x)

        u_NN = u_nn_test.reshape(x[:, 1].shape)
        if(torch.cuda.is_available()):
            u_NN = u_NN.cpu().detach().numpy()
        else:
            u_NN = u_NN.detach().numpy()

        # u_NN = u_NN.reshape((domain.num_test_samples_shapes[1], domain.num_test_samples_shapes[0]))
        # fig = plt.figure()
        # gs0 = gridspec.GridSpec(10,1)
        # gs0.update(top=0.95, bottom=0.45, left=0.15, right=0.85, wspace=0)
        # ax = plt.subplot(gs0[:, :], label='u')

        # h = ax.imshow(u_NN, interpolation='nearest', cmap='rainbow',
        #                     extent=[0, 5, -1, 1],
        #                     origin='lower',
        #                     aspect='auto', label='u')

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(h, cax=cax)

        # ax.set_title('PINN solution', fontsize=10)
        # ax.set_box_aspect(aspect = (2.5, 1.25, 1.0))
        # fig.colorbar(h, shrink=0.5)
        # ax.view_init(35, 75)

        # plt.savefig(fig_name)

        pixels_shape = (
            domain.num_test_samples_shapes[1], domain.num_test_samples_shapes[0])
        u_NN = u_NN.reshape(pixels_shape)

        fig = plt.figure()
        gs0 = gridspec.GridSpec(1, 1)
        gs0.update(top=0.95, bottom=0.8, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :], label='u')
        h = ax.imshow(u_NN,
                      interpolation='nearest',
                      cmap='rainbow',
                      extent=[0, 12, -1, 1],
                      origin='lower',
                      aspect='auto', label='u')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)

        ax.set_title('PINN solution', fontsize=10)
        plt.savefig(fig_name+'.jpg')

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        if(torch.cuda.is_available()):
            h = ax.plot_surface(x[:, 0].cpu().detach().numpy().reshape(pixels_shape), x[:, 1].cpu(
            ).detach().numpy().reshape(pixels_shape), u_NN,  cmap='rainbow')
            ax.contour(x[:, 0].cpu().detach().numpy().reshape(pixels_shape), x[:, 1].cpu().detach().numpy(
            ).reshape(pixels_shape), u_NN, 12, offset=-1, linestyles="solid", alpha=0.5, cmap='rainbow')
        else:
            h = ax.plot_surface(x[:, 0].detach().numpy().reshape(
                pixels_shape), x[:, 1].detach().numpy().reshape(pixels_shape), u_NN,  cmap='rainbow')
            ax.contour(x[:, 0].detach().numpy().reshape(pixels_shape), x[:, 1].detach().numpy().reshape(
                pixels_shape), u_NN, 12, offset=-1, linestyles="solid", alpha=0.5, cmap='rainbow')

        ax.set_title('PINN solution', fontsize=10)
        ax.set_box_aspect(aspect=(2.5, 1.25, 1.0))
        fig.colorbar(h, shrink=0.5)
        ax.view_init(35, 75)

        fig.tight_layout()
        plt.savefig(fig_name+"_proj.jpg")




class BC_KleinGordon1D_plus_time(BCEnforceArchitecture):
    def __init__(self, start_point, end_point):
        super(BC_KleinGordon1D_plus_time, self).__init__()

        self.startPoint_x = torch.tensor(start_point[0])  # xmin
        self.startPoint_y = torch.tensor(start_point[1])  # ymin

        # end point is upper right point of the bounding box
        self.endPoint_x = torch.tensor(end_point[0])    # xmax
        self.endPoint_y = torch.tensor(end_point[1])    # xmin
  

    def bc_fun(self, coords):
        return coords[:,1] * torch.cos(coords[:, 0])


    def lifting(self, coords):
        x = coords[:,0]
        y = coords[:,1]

        return ((y - self.startPoint_y) * (self.endPoint_y - y) * (x - self.startPoint_x) *  (self.endPoint_x - x))


    # simple trick due to tailing 
    def apply_bc(self, net_out, coords):
        out = self.bc_fun(coords) + (net_out[:,0]*self.lifting(coords))
        return out




if __name__ == '__main__':

    pass
    # pde = KleinGordon()

    # # x = np.linspace(0, 1, 100, endpoint=True)
    # # x = x.reshape((-1, 1))
    # # x = torch.tensor(x)

    # # exact = pde.analytical_sol(x)
    # # plt.plot(x.detach().numpy(), exact.detach().numpy())
    # # plt.ylabel('Sol')
    # # plt.show()

    # train_samples = np.array([10, 10])
    # test_samples = np.array([50, 50])
    # start_point = np.array([0., 0.])
    # end_point = np.array([1., 1.])

    # dim = 2
    # penaltyPara = 1e2
