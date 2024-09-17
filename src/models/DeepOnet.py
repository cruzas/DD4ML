import torch
import torch.nn as nn

class DeepOnet(nn.Module):

    def __init__(self, branch_net, trunk_net, trunk_net_input, exact_boundary=None):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.branch_net = branch_net
        self.branch_net.init_params()

        self.trunk_net = trunk_net
        self.trunk_net.init_params()
        
        self.trunk_net_input = trunk_net_input.to(self.device)
        self.exact_boundary = exact_boundary

        self.num_params_branch = sum(p.numel() for p in self.branch_net.parameters() if p.requires_grad)
        self.num_params_trunk = sum(p.numel() for p in self.trunk_net.parameters() if p.requires_grad)

    def forward(self, x):

        branch_outputs_net = self.branch_net(x[0])
        trunk_outputs_net = self.trunk_net.act(self.trunk_net(self.trunk_net_input))

        u_pred  = branch_outputs_net @ trunk_outputs_net.T 
        if self.exact_boundary is not None:
            u_pred = self.exact_boundary(u_pred)

        return u_pred

    def get_avg(self):
        avg = []
        if hasattr(self.branch_net, 'avg'):
            avg.append(self.branch_net.avg)
        else:
            avg.append(torch.zeros(self.num_params_branch, dtype=torch.get_default_dtype()).to(self.device))
        if hasattr(self.trunk_net, 'avg'):
            avg.append(self.trunk_net.avg)
        else:
            avg.append(torch.zeros(self.num_params_trunk, dtype=torch.get_default_dtype()).to(self.device))

        return torch.cat(avg)

    def get_num_layers(self):
        return self.branch_net.get_num_layers() + self.trunk_net.get_num_layers()

    def extract_trainable_params(self, sbd_id, num_subdomains, overlap_width=0):
        num_layers_branch, num_layers_trunk = self.branch_net.get_num_layers(), self.trunk_net.get_num_layers()
        num_subdomains_branch = max(round(num_subdomains * num_layers_branch / (num_layers_branch + num_layers_trunk)), 1)  # minimum num_layers_per_subdomains=1
        num_subdomains_trunk = num_subdomains - num_subdomains_branch
        if sbd_id < num_subdomains_branch:
            trainable_params = self.branch_net.extract_trainable_params(sbd_id, num_subdomains_branch, overlap_width)
        else:
            trainable_params =  self.trunk_net.extract_trainable_params(sbd_id - num_subdomains_branch, num_subdomains_trunk, overlap_width)
        return trainable_params

    def extract_coarse_trainable_params(self, num_subdomains, overlap_width=0):
        num_layers_branch, num_layers_trunk = self.branch_net.get_num_layers(), self.trunk_net.get_num_layers()
        num_subdomains_branch = max(round(num_subdomains * num_layers_branch / (num_layers_branch + num_layers_trunk)), 1)  # minimum num_layers_per_subdomains=1
        num_subdomains_trunk = num_subdomains - num_subdomains_branch
        return self.branch_net.extract_coarse_trainable_params(num_subdomains_branch, overlap_width) + self.trunk_net.extract_coarse_trainable_params(num_subdomains_trunk, overlap_width)

    def extract_sbd_quantity(self, sbd_id, num_subdomains, all_tensors, overlap_width=0):
        num_layers_branch, num_layers_trunk = self.branch_net.get_num_layers(), self.trunk_net.get_num_layers()
        num_subdomains_branch = max(round(num_subdomains * num_layers_branch / (num_layers_branch + num_layers_trunk)), 1)  # minimum num_layers_per_subdomains=1
        num_subdomains_trunk = num_subdomains - num_subdomains_branch
        if sbd_id < num_subdomains_branch:
            trainable_params = self.branch_net.extract_sbd_quantity(sbd_id, num_subdomains_branch, all_tensors, overlap_width)
        else:
            trainable_params =  self.trunk_net.extract_sbd_quantity(sbd_id - num_subdomains_branch, num_subdomains_trunk, all_tensors, overlap_width)
        return trainable_params

    def print_decomposition(self, num_subdomains):
        num_layers_branch, num_layers_trunk = self.branch_net.get_num_layers(), self.trunk_net.get_num_layers()
        num_subdomains_branch = max(round(num_subdomains * num_layers_branch / (num_layers_branch + num_layers_trunk)), 1)  # minimum num_layers_per_subdomains=1
        num_subdomains_trunk = num_subdomains - num_subdomains_branch
        print(" - - -  branch net  - - -")
        self.branch_net.print_decomposition(num_subdomains_branch)
        print(" - - -  trunk net  - - -")
        self.trunk_net.print_decomposition(num_subdomains_trunk)
        return 0

if __name__ == '__main__':
    print("Not implemented... ")
    exit(0)



