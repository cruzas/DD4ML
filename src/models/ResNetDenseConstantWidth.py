import torch
import torch.nn as nn
import math
from torchsummary import summary
import torch.nn.init as init
import torch.distributed as dist


class MLHierarchyResNets(object): 
    def __init__(self, num_levels):
        self.num_levels = num_levels
        self.nets = []


    def build(self, inputs=2, outputs=1, width=20, hiddenlayers_coarse=2, T=1.0, use_adaptive_activation=False): 
        hiddenlayers = hiddenlayers_coarse
        dt = T/(hiddenlayers-1)

        for l in range(0, self.num_levels):

            print("level:  ", l, "   layers:   ", hiddenlayers, "   dt:  ", dt, "  T:  ", T)

            self.nets.append(ResNetDenseConstantWidth(inputs=inputs, outputs=outputs, hiddenlayers=hiddenlayers, width=width, dt=dt, use_adaptive_activation=use_adaptive_activation))
            hiddenlayers = (hiddenlayers*2)-1
            dt = T/(hiddenlayers-1)

        return self.nets



    def prolong_params(self, coarse_net, fine_net): 

        # first layer
        fine_net.linear_in.weight.data.copy_(coarse_net.linear_in.weight.data)
        fine_net.linear_in.bias.data.copy_(coarse_net.linear_in.bias.data)


        # last layer
        fine_net.linear_out.weight.data.copy_(coarse_net.linear_out.weight.data)
        fine_net.linear_out.bias.data.copy_(coarse_net.linear_out.bias.data)

        for s_fine in range(0, len(fine_net.stage)): 

            if(s_fine%2==0):
                s_coarse = int(s_fine/2)

                fine_net.stage[s_fine].layer.weight.data.copy_(coarse_net.stage[s_coarse].layer.weight.data)
                fine_net.stage[s_fine].layer.bias.data.copy_(coarse_net.stage[s_coarse].layer.bias.data)      

                if(fine_net.use_adaptive_activation): 
                    fine_net.stage[s_fine].a_adaptive_fun.data.copy_(coarse_net.stage[s_coarse].a_adaptive_fun.data)       

            else:
                s_coarse_left = int((s_fine-1)/2)
                s_coarse_right = int((s_fine+1)/2)
                
                fine_net.stage[s_fine].layer.weight.data.copy_(0.5*coarse_net.stage[s_coarse_left].layer.weight.data + 0.5*coarse_net.stage[s_coarse_right].layer.weight.data)
                fine_net.stage[s_fine].layer.bias.data.copy_(0.5*coarse_net.stage[s_coarse_left].layer.bias.data + 0.5*coarse_net.stage[s_coarse_right].layer.bias.data)                
                
                if(fine_net.use_adaptive_activation):
                    fine_net.stage[s_fine].a_adaptive_fun.data.copy_(0.5*coarse_net.stage[s_coarse_left].a_adaptive_fun.data + 0.5*coarse_net.stage[s_coarse_right].a_adaptive_fun.data)                





    def prolong_correction(self, coarse_net_init, coarse_net_sol, fine_net): 

        # first layer
        fine_net.linear_in.weight.data.add_(coarse_net_sol.linear_in.weight.data - coarse_net_init.linear_in.weight.data)
        fine_net.linear_in.bias.data.add_(coarse_net_sol.linear_in.bias.data - coarse_net_init.linear_in.bias.data)


        # last layer
        fine_net.linear_out.weight.data.add_(coarse_net_sol.linear_out.weight.data - coarse_net_init.linear_out.weight.data)
        fine_net.linear_out.bias.data.add_(coarse_net_sol.linear_out.bias.data - coarse_net_init.linear_out.bias.data)


        for s_fine in range(0, len(fine_net.stage)): 

            if(s_fine%2==0):
                s_coarse = int(s_fine/2)

                fine_net.stage[s_fine].layer.weight.data.add_(coarse_net_sol.stage[s_coarse].layer.weight.data - coarse_net_init.stage[s_coarse].layer.weight.data)
                fine_net.stage[s_fine].layer.bias.data.add_(coarse_net_sol.stage[s_coarse].layer.bias.data - coarse_net_init.stage[s_coarse].layer.bias.data)      

                if(fine_net.use_adaptive_activation): 
                    fine_net.stage[s_fine].a_adaptive_fun.data.add_(coarse_net_sol.stage[s_coarse].a_adaptive_fun.data - coarse_net_init.stage[s_coarse].a_adaptive_fun.data)       

            else:
                s_coarse_left = int((s_fine-1)/2)
                s_coarse_right = int((s_fine+1)/2)
                
                fine_net.stage[s_fine].layer.weight.data.add_(0.5*(coarse_net_sol.stage[s_coarse_left].layer.weight.data-coarse_net_init.stage[s_coarse_left].layer.weight.data) + 0.5*(coarse_net_sol.stage[s_coarse_right].layer.weight.data - coarse_net_init.stage[s_coarse_right].layer.weight.data))
                fine_net.stage[s_fine].layer.bias.data.add_(0.5*(coarse_net_sol.stage[s_coarse_left].layer.bias.data-coarse_net_init.stage[s_coarse_left].layer.bias.data) + 0.5*(coarse_net_sol.stage[s_coarse_right].layer.bias.data - coarse_net_init.stage[s_coarse_right].layer.bias.data))         
                
                if(fine_net.use_adaptive_activation):
                    fine_net.stage[s_fine].a_adaptive_fun.data.add_(0.5*(coarse_net_sol.stage[s_coarse_left].a_adaptive_fun.data-coarse_net_init.stage[s_coarse_left].a_adaptive_fun.data) + 0.5*(coarse_net_sol.stage[s_coarse_right].a_adaptive_fun.data - coarse_net_init.stage[s_coarse_right].a_adaptive_fun.data))




    def init_coarse_params_injection(self, coarse_net, fine_net): 

        # first layer
        coarse_net.linear_in.weight.data.copy_(fine_net.linear_in.weight.data)
        coarse_net.linear_in.bias.data.copy_(fine_net.linear_in.bias.data)

        # last layer
        coarse_net.linear_out.weight.data.copy_(fine_net.linear_out.weight.data)
        coarse_net.linear_out.bias.data.copy_(fine_net.linear_out.bias.data)


        for s_fine in range(0, len(fine_net.stage)): 

            if(s_fine%2==0):
                s_coarse = int(s_fine/2)

                # if(dist.get_rank()):
                #     print("s_fine ", s_fine, " s_coarse ", s_coarse)

                coarse_net.stage[s_coarse].layer.weight.data.copy_(fine_net.stage[s_fine].layer.weight.data)
                coarse_net.stage[s_coarse].layer.bias.data.copy_(fine_net.stage[s_fine].layer.bias.data)      

                if(coarse_net.use_adaptive_activation): 
                    coarse_net.stage[s_coarse].a_adaptive_fun.data.copy_(fine_net.stage[s_fine].a_adaptive_fun.data)       





    def init_coarse_params_weighted_restriction(self, coarse_net, fine_net): 

        # first layer
        coarse_net.linear_in.weight.data.copy_(fine_net.linear_in.weight.data)
        coarse_net.linear_in.bias.data.copy_(fine_net.linear_in.bias.data)

        # last layer
        coarse_net.linear_out.weight.data.copy_(fine_net.linear_out.weight.data)
        coarse_net.linear_out.bias.data.copy_(fine_net.linear_out.bias.data)


        for s_fine in range(0, len(fine_net.stage)): 

            if(s_fine%2==0):
                s_coarse = int(s_fine/2)


                if(s_fine==0 or s_fine==(len(fine_net.stage)-1)):
                    # if(dist.get_rank()):
                    #     print("s_fine ", s_fine, " s_coarse ", s_coarse)                    

                    coarse_net.stage[s_coarse].layer.weight.data.copy_(fine_net.stage[s_fine].layer.weight.data)
                    coarse_net.stage[s_coarse].layer.bias.data.copy_(fine_net.stage[s_fine].layer.bias.data)      

                    if(coarse_net.use_adaptive_activation): 
                        coarse_net.stage[s_coarse].a_adaptive_fun.data.copy_(fine_net.stage[s_fine].a_adaptive_fun.data)       

                else:
                    # if(dist.get_rank()):
                    #     print(" s_coarse ", s_coarse,  "s_fine ", s_fine, " s_fine_right ", s_fine-1, " s_fine_right ", s_fine+1)

                    coarse_net.stage[s_coarse].layer.weight.data.copy_(0.5*fine_net.stage[s_fine].layer.weight.data + 0.25*fine_net.stage[s_fine-1].layer.weight.data + 0.25*fine_net.stage[s_fine+1].layer.weight.data)
                    coarse_net.stage[s_coarse].layer.bias.data.copy_(0.5*fine_net.stage[s_fine].layer.bias.data + 0.25*fine_net.stage[s_fine-1].layer.bias.data+ 0.25*fine_net.stage[s_fine+1].layer.bias.data)                
                    
                    if(coarse_net.use_adaptive_activation):
                        coarse_net.stage[s_coarse].a_adaptive_fun.data.copy_(0.5*fine_net.stage[s_fine].a_adaptive_fun.data + 0.25*fine_net.stage[s_fine-1].a_adaptive_fun.data+ 0.25*fine_net.stage[s_fine+1].a_adaptive_fun.data)                






class ResNetBlock(nn.Module):
    def __init__(self, in_size, out_size, dt, act_fun="relu", use_adaptive_activation=True):
        super(ResNetBlock, self).__init__()

        self.dt                         = dt
        self.layer                      =  nn.Linear(in_features=in_size, out_features=out_size, bias=True)        
        self.use_adaptive_activation    = use_adaptive_activation

        if(act_fun=="relu"):
            self.act = nn.ReLU(inplace=True)
        elif(act_fun=="tanh"):
            self.act = nn.Tanh()   
        elif(act_fun=="sigmoid"):
            self.act = nn.Sigmoid()               
        else:
            print("wrong choice of activation function")         
            exit(0)

        if(self.use_adaptive_activation):
            # value good for tanh, TODO:: fix for rest 
            self.n                     = 10
            self.a_adaptive_fun        = nn.Parameter(torch.tensor(1./self.n), requires_grad=True)


class ResNetDenseConstantWidth(nn.Module):

    def __init__(self, inputs, outputs, hiddenlayers, width, dt=1.0, act_fun="tanh", use_adaptive_activation=True):
        super(ResNetDenseConstantWidth, self).__init__()

        self.use_adaptive_activation = use_adaptive_activation

        self.linear_in  = nn.Linear(inputs, width)
        self.stage      = self.get_stage(hiddenlayers, width, dt, act_fun)
        self.linear_out = nn.Linear(width, outputs, bias=True)

        # self.a_adaptive_fun     = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        # self.weight_data        = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        # self.weight_residual    = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.dt = dt

        self.layer_ids = []
        for name, param in self.named_parameters():
            name_str = name.split(".")
            
            if(name_str[0]=="linear_in"):
                self.layer_ids.append(0)
            elif(name_str[0]=="stage"):
                self.layer_ids.append(int(name_str[1])+1)
            elif(name_str[0]=="linear_out"):
                self.layer_ids.append(len(self.stage)+1)                        

        self.num_layers = max(self.layer_ids)+1

        if(act_fun=="relu"):
            self.act = nn.ReLU(inplace=True)
        elif(act_fun=="tanh"):
            self.act = nn.Tanh()   
        elif(act_fun=="sigmoid"):
            self.act = nn.Sigmoid()               
        else:
            print("wrong choice of activation function")         
            exit(0)

        # self.init_params()

    def get_stage(self, depth, width, dt, act_fun):
        
        layers = []
        # layers associated with time-discretization of ODE
        for stride in range(0, depth):
            layers.append(ResNetBlock(in_size=width, out_size=width, dt=dt, act_fun=act_fun, use_adaptive_activation=self.use_adaptive_activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.linear_in(x))

        for i, block in enumerate(self.stage):
            xx = out
            # out = block.act(10*self.a_adaptive_fun*block.layer(out))
            if(self.use_adaptive_activation):
                out = block.act(block.n*block.a_adaptive_fun*block.layer(out))
            else:
                out = block.act(block.layer(out))

            out = (block.dt * out) + xx

        y = self.linear_out(out)

        return y            



    def get_num_layers(self):
        return len(self.stage) + 2

    def get_avg(self):
        return self.avg

    # # # # # # # # # # # # # # # Contains overlapping version # # # # # # # # # # # # # 
    def extract_trainable_params(self, sbd_id, num_subdomains, overlap_width=0):
        num_layers_per_subdomain = round(self.num_layers/num_subdomains)
        stp = max(0, sbd_id*num_layers_per_subdomain - overlap_width)
        edp = min(self.num_layers, (sbd_id + 1)*num_layers_per_subdomain + overlap_width)
        subdomain_ids = list(range(stp, edp))

        trainable_params=[]
        trainable_index=[]
        for index, p in enumerate(self.parameters()):
            if self.layer_ids[index] in subdomain_ids:
                trainable_params.append(p)
                trainable_index.append(index)

        self.avg = []
        for index, p in enumerate(self.parameters()): 
            if index in trainable_index:
                self.avg.append(torch.ones_like(p.view(-1)))
            else:
                self.avg.append(torch.zeros_like(p.view(-1)))
        self.avg = torch.cat(self.avg)

        return trainable_params

    def extract_coarse_trainable_params(self, num_subdomains, overlap_width=0):
        num_layers_per_subdomain = round(self.num_layers/num_subdomains)
        coarsedomain_ids = []
        for sbd_id in range(num_subdomains):
            stp = max(0, sbd_id*num_layers_per_subdomain - overlap_width)
            coarsedomain_ids.append(stp)

        trainable_params=[]
        for index, p in enumerate(self.parameters()):
            if self.layer_ids[index] in coarsedomain_ids:
                trainable_params.append(p)

        return trainable_params

    def extract_multilevel_trainable_params(self, num_subdomains, level=1):
        num_coarsedomain = int(num_subdomains / (2 ** (level - 1)))
        num_layers_per_subdomain = round(self.num_layers/num_coarsedomain)
        coarse_ids = [num_layers_per_subdomain * i for i in range(num_coarsedomain)]

        trainable_params=[]
        for index, p in enumerate(self.parameters()): 
            if self.layer_ids[index] in coarse_ids:
                trainable_params.append(p)

        return trainable_params

    # # # # # # # # # # # # # # # Contains overlapping version # # # # # # # # # # # # # 
    def extract_sbd_quantity(self, sbd_id, num_subdomains, all_tensors, overlap_width=0):
        num_layers_per_subdomain = round(self.num_layers/num_subdomains)
        stp = max(0, sbd_id*num_layers_per_subdomain - overlap_width)
        edp = min(self.num_layers, (sbd_id + 1)*num_layers_per_subdomain + overlap_width)
        subdomain_ids = list(range(stp, edp))

        trainable_params=[]
        for index, p in enumerate(all_tensors):
            if self.layer_ids[index] in subdomain_ids:
                trainable_params.append(p)

        return trainable_params

    def print_decomposition(self, num_subdomains):

        num_layers_per_subdomain = round(self.num_layers/num_subdomains)

        for sbd_id in range(0, num_subdomains):
            for index, p in enumerate(self.parameters()): 
                index_layer = int(self.layer_ids[index]/num_layers_per_subdomain)
                # print("index_layer ", index_layer)
                
                # if(sbd_id==index_layer or (index_layer>(num_layers_per_subdomain) and sbd_id==num_subdomains-1)):
                if(sbd_id==index_layer or (index_layer>=num_subdomains and sbd_id==num_subdomains-1)):
                    print("layer  ", index,  "layer id. ", self.layer_ids[index],  "sbd_id:  ", sbd_id)


        return 0



    def init_params(self):
        self.apply(self._init_layers)    


    def _init_layers(self, m):
        classname = m.__class__.__name__

        if isinstance(m, nn.Linear):
            if m.weight is not None:
                init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)




if __name__ == '__main__':

    torch.manual_seed(0)
    net = ResNetDenseConstantWidth(inputs=1, outputs=1, hiddenlayers=6, width=10, dt=0.5)
    net.init_params()

    sbd_id = 1
    num_subdomains = 4
    overlap_width = 0
    params_subdomain = net.extract_trainable_params(sbd_id, num_subdomains, overlap_width)
    # print('subdomain: \n', params_subdomain)
    # summary(net, (1,))

    # check the overlapping region
    # import matplotlib.pyplot as plt
    # lists = []
    # for i in range(num_subdomains):
    #     params_subdomain = net.extract_trainable_params(i, num_subdomains, overlap_width)
    #     lists.append(net.avg)
    # avgs = sum(lists)
    # plt.plot(range(len(avgs)), avgs)
    # plt.show()
