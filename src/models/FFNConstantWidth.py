import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init


class FFNConstantWidth(nn.Module):

    def __init__(self, inputs, outputs, hiddenlayers, width, act_fun="relu", use_adaptive_activation=True):
        super(FFNConstantWidth, self).__init__()

        self.use_adaptive_activation = use_adaptive_activation

        self.linearIn = nn.Linear(inputs, width)

        # layer definitions
        self.linear          = nn.ModuleList()
        self.ada_act_funs    = []
        self.n               = 10
        for _ in range(hiddenlayers):
            self.linear.append(nn.Linear(width, width))
            self.ada_act_funs.append(nn.Parameter(torch.tensor(1./self.n), requires_grad=True))

        self.linearOut = nn.Linear(width, outputs, bias=True)

        self.layer_ids = []
        for name, param in self.named_parameters():
            name_str = name.split(".")
            
            if(name_str[0]=="linearIn"):
                self.layer_ids.append(0)
            elif(name_str[0]=="linear"):
                self.layer_ids.append(int(name_str[1])+1)
            elif(name_str[0]=="linearOut"):
                self.layer_ids.append(len(self.linear)+1)                        

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


    def forward(self, x):
        xx = self.act(self.linearIn(x))

        i=0
        for layer in self.linear:
            if(self.use_adaptive_activation):
                xx = self.act(self.n*self.ada_act_funs[i]*layer(xx))
                i+=1
            else:
                xx = self.act(layer(xx))

        y = self.linearOut(xx)

        return y


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
                init.normal_(m.weight)
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


    net = FFNConstantWidth(inputs=1, outputs=1, hiddenlayers=4, width=10)
    net.init_params()


    summary(net, (1,))


