import numpy as np
import torch
import pyDOE
from torch.utils.data import Dataset


class Domain1D(Dataset):

    def __init__(self, start_point, end_point, num_train_samples, num_test_samples, batch_size=None, sampler="uniform", withBC=False):
        self.dim = 1
        self.startPoint = start_point
        self.endPoint = end_point
        self.sampler = sampler
        self.withBC = withBC
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        
        self.create_1d_domain()

        if(batch_size is None):
            self.train_loader = torch.utils.data.DataLoader(self.xTrainInterior, batch_size=num_train_samples, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(self.xTestInterior, batch_size=num_test_samples, shuffle=False)

        else:
            self.train_loader = torch.utils.data.DataLoader(self.xTrainInterior, batch_size=batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(self.xTestInterior, batch_size=batch_size, shuffle=False)



    def create_1d_domain(self):
        self.volDomain = torch.tensor([[self.endPoint - self.startPoint]])
        self.volBC = torch.tensor([[2.]])

        if(self.sampler == "uniform"):
            self.xTrainInterior = self.uniform_distribution(self.num_train_samples, self.withBC)
        elif(self.sampler == "latin_hypercube"):
            self.xTrainInterior = self.latin_hypercube_distribution(self.num_train_samples)
        else:
            print("sampling strategy not valid, proceed with uniform distribution")
            self.xTrainInterior = self.uniform_distribution(self.num_train_samples, self.withBC)            


        self.xTestInterior = self.uniform_distribution(self.num_test_samples, withBC=True)            

        self.leftBC = torch.tensor(np.array([[self.startPoint]]))
        self.rightBC = torch.tensor(np.array([[self.endPoint]]))

        self.xTrain = self.create_domain_set(
            self.xTrainInterior, self.leftBC, self.rightBC)

        self.xBC = torch.cat((self.leftBC, self.rightBC), dim=0)

        self.xTest = self.uniform_distribution(self.num_test_samples, True)


    def uniform_distribution(self, num_samples, withBC):

        if withBC:
            x = np.linspace(self.startPoint, self.endPoint,
                            num_samples, endpoint=True)
            x = x.reshape((-1, 1))            
            x = torch.tensor(x)
        elif not withBC:
            x = np.linspace(self.startPoint, self.endPoint,
                            num_samples+2, endpoint=True)
            x = x.reshape((-1, 1))            
            x = torch.tensor(x[1:-1, :])

        return x

    def latin_hypercube_distribution(self, num_samples):
        x = pyDOE.lhs(1, samples=num_samples)
        x = x[:, 0]
        x = x.reshape((-1, 1))
        x = torch.tensor(x)
        x = x * self.volDomain + self.startPoint

        return x

    def create_domain_set(self, x, bc_left, bc_right):
        x = torch.cat((bc_left, x), dim=0)
        x = torch.cat((x, bc_right), dim=0)
        return x


    def __len__(self):
        return len(self.xTrainInterior)


    def __getitem__(self, idx):
        sample = self.xTrainInterior[idx]
        return sample        



class BC_1D(object):

    def __init__(self, x_start, x_end, bc_val_start, bc_val_end):
        self.x_start = x_start
        self.x_end = x_end

        self.bc_val_start = bc_val_start
        self.bc_val_end = bc_val_end

    def analytical(self, x):
        # if you know exact solution on BC, use this one
        # out = torch.sin(pi * x)

        # equation of line
        out = (self.bc_val_end - self.bc_val_start) / (self.x_end - self.x_start) * \
            (x - self.x_start) + self.bc_val_start

        return out

    def apply_bc(self, x, net_out):
        out = self.analytical(
            x) + (((x - self.x_start) * (self.x_end - x)) * net_out)
        return out





if __name__ == '__main__':

    domain = Domain1D(start_point=0, end_point=1, num_train_samples=100, num_test_samples=20, batch_size=20, withBC=False)

    print(len(domain))

    item = domain.__getitem__(1)
    print("getitem: ", item)    

    for batch_idx, (inputs) in enumerate(domain.trainloader):
        print("batch_idx: ", batch_idx, "   inputs:  ", inputs)


    for batch_idx, (inputs) in enumerate(domain.testloader):
        print("batch_idx: ", batch_idx, "   inputs:  ", inputs)





