import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init
import sys
class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, act_fun="relu", use_adaptive_activation=True):
        super().__init__()
        self.layer = nn.Conv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride)
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

    def forward(self, x):
        if(self.use_adaptive_activation):
            x = self.act(self.n * self.a_adaptive_fun * self.layer(x))
        else:
            x = self.act(self.layer(x))
        return x

class FCNBlock(nn.Module):
    def __init__(self, inputs, outputs, act_fun="relu", use_adaptive_activation=True):
        super().__init__()
        self.layer = nn.Linear(inputs, outputs)
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

    def forward(self, x):
        if(self.use_adaptive_activation):
            x = self.act(self.n * self.a_adaptive_fun * self.layer(x))
        else:
            x = self.act(self.layer(x))
        return x

class CNN3D(nn.Module):

    def __init__(self, inputs, outputs, hiddenlayers=7, width=40, act_fun="relu", use_adaptive_activation=True):
        super().__init__()
        ''' total number of layers = hiddenlayers (Conv3D) + 3 (Linear) '''

        num_fc_layers = 3
        num_conv_layers = hiddenlayers - num_fc_layers
        self.use_adaptive_activation = use_adaptive_activation
        self.kernel_size = (3, 3, 3)
        self.stride = 2

        channels_list = [inputs, width] + [int(width / 2 * (2 ** (conv_layer_index + 1) + 1)) for conv_layer_index in range(num_conv_layers-1)]
        features_list = [channels_list[-1]] + [2 * width] * (num_fc_layers-1) + [outputs]
        
        self.conv = nn.Sequential(*[Conv3DBlock(channels_list[conv_layer_index], channels_list[conv_layer_index+1], kernel_size=self.kernel_size, stride=self.stride, act_fun=act_fun, use_adaptive_activation=use_adaptive_activation) for conv_layer_index in range(num_conv_layers)])
        self.linear = nn.Sequential(*[FCNBlock(features_list[fc_layer_index], features_list[fc_layer_index+1], act_fun=act_fun, use_adaptive_activation=use_adaptive_activation) for fc_layer_index in range(num_fc_layers)])
        self.flat = nn.Flatten()

        self.layer_ids = []
        for name, param in self.named_parameters():
            name_str = name.split(".")
            if(name_str[0]=="conv"):
                self.layer_ids.append(int(name_str[1]))
            elif(name_str[0]=="linear"):
                self.layer_ids.append(int(name_str[1]) + len(self.conv))
        self.num_layers = max(self.layer_ids)+1

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear(x)
        return x


    def get_num_layers(self):
        return len(self.linear) + 2

    def get_avg(self):
        return self.avg

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



    def init_params(self):
        self.apply(self._init_layers)    


    def _init_layers(self, m):
        classname = m.__class__.__name__

        if isinstance(m, nn.Linear):
            if m.weight is not None:
                init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv3d):
            if m.weight is not None:
                init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

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



if __name__ == '__main__':


    net = CNN3D(inputs=20, outputs=10, hiddenlayers=7, width=40)
    net.init_params()
    x = torch.ones((1, 20, 33, 33, 33))
    print(net(x).shape)

    sbd_id = 1
    num_subdomains = 4
    overlap_width = 0
    net.print_decomposition(num_subdomains)
    params_subdomain = net.extract_trainable_params(sbd_id, num_subdomains, overlap_width)
    print('subdomain: \n', params_subdomain)
