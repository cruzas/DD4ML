import numpy as np
import copy
import sys
import torch
import pyDOE  # py -3.10 -m pip install pyDOE
import random
from torch.utils.data import Dataset
import skopt # py -3.10 -m pip install scikit-optimize
import torch.distributed as dist
from dataloaders import GeneralizedDistributedDataLoader


class RectangleDomain2D(object):

    def __init__(self, start_point, end_point, num_train_samples, num_test_samples, use_BC=False, sampling_strategy="latin_hypercube", ref_strategy="RAD", x_interior=None, xBC_train=None, model_structure=None):

        self.model_structure = model_structure
        self.dim = 2
        self.use_BC = use_BC
        
        # start point is lower left point of the bounding box
        self.startPoint_x = start_point[0]  # xmin
        self.startPoint_y = start_point[1]  # ymin

        # end point is upper right point of the bounding box
        self.endPoint_x = end_point[0]    # xmax
        self.endPoint_y = end_point[1]    # yax
        

        self.vol_domain = torch.tensor(
            [[(self.endPoint_x - self.startPoint_x)*(self.endPoint_y - self.startPoint_y)]])


        self.sampling_strategy = sampling_strategy
        self.ref_strategy      = ref_strategy

            
        self.num_train_samples_shapes   = num_train_samples
        self.num_test_samples_shapes    = num_test_samples


        if(x_interior is not None and xBC_train is not None):
            self.x_interior = x_interior
            self.xBC_train  = xBC_train

        else:
            self.x_interior, self.xBC_train = self.create_RectangleDomain2D_train(num_train_samples, num_train_samples[0]*num_train_samples[1])
            

        self.create_RectangleDomain2D_test(num_test_samples)


        # train sampler
        if(self.use_BC):
            self.x_in_bc = torch.cat((self.x_interior, self.xBC_train), dim=0)

            if dist.is_initialized():
                #our stuff
                raise NotImplementedError
            else:
                sampler_train = torch.utils.data.sampler.BatchSampler(
                                torch.utils.data.sampler.RandomSampler(self.x_in_bc),
                                batch_size=int(len(self.x_in_bc)),
                                drop_last=False)

                self.train_loader = torch.utils.data.DataLoader(self.x_in_bc, sampler=sampler_train)
        else:
            if dist.is_initialized():
                #our stuff
                # raise NotImplementedError
                self.train_loader = GeneralizedDistributedDataLoader(model_structure=self.model_structure, dataset=self.x_interior, batch_size=int(len(self.x_interior)), shuffle=False, num_workers=0, pin_memory=True)
            else:
                sampler_train = torch.utils.data.sampler.BatchSampler(
                                torch.utils.data.sampler.RandomSampler(self.x_interior),
                                batch_size=int(len(self.x_interior)),
                                drop_last=False)

                self.train_loader = torch.utils.data.DataLoader(self.x_interior, sampler=sampler_train)


            # print("len(self.x_interior) ", len(self.x_interior))


        
        # test sampler
        if dist.is_initialized():
            #our stuff
            self.test_loader = GeneralizedDistributedDataLoader(model_structure=self.model_structure, dataset=self.x_test, batch_size=int(len(self.x_test)), shuffle=False, num_workers=0, pin_memory=True)
        else:
            sampler_test =  torch.utils.data.sampler.BatchSampler(
                            torch.utils.data.sampler.RandomSampler(self.x_test),
                            batch_size=int(len(self.x_test)),
                            drop_last=False)

            self.test_loader = torch.utils.data.DataLoader(self.x_test, sampler=sampler_test)


    def adaptivelly_refine(self, net, pde):

        self.use_cuda = torch.cuda.is_available()




        if(self.ref_strategy=="RAR"):
            m = 100

            x_new, x_BC_new = self.create_RectangleDomain2D_train(self.num_train_samples_shapes, m*m)

            if self.use_cuda:
                x_new = x_new.cuda()
            x_new.requires_grad = True            

            net_out = net(x_new)

            if(pde.bc_exact is not None):
                net_out = pde.bc_exact.apply_bc(net_out, x_new)


            residual = pde.residual(net_out, x_new)
            residual = torch.abs(residual)
            residual = residual.cpu().detach().numpy()
            x_new    = x_new.cpu().detach().numpy()

            # print("x_new ", x_new)
            # print("residual ", residual)


            sorted_index_residual   = np.argsort(residual)
            sorted_residual         = residual[sorted_index_residual]
            sorted_points           = x_new[sorted_index_residual]

            # np.set_printoptions(threshold=sys.maxsize)

            # print("sorted_residual ", sorted_residual)
            # print("sorted_points ",   sorted_points)


            m_max_points            = sorted_points[-m:]

            # print("m_max_points ", m_max_points)

            self.x_interior = torch.cat((self.x_interior, torch.tensor(m_max_points)), dim=0)


            if(self.use_BC):
                self.x_in_bc = torch.cat((self.x_interior, self.xBC_train), dim=0)
                if dist.is_initialized():
                    #our stuff
                    raise NotImplementedError
                else:
                    sampler_train = torch.utils.data.sampler.BatchSampler(
                                    torch.utils.data.sampler.RandomSampler(self.x_in_bc),
                                    batch_size=int(len(self.x_in_bc)),
                                    drop_last=False)

                    self.train_loader = torch.utils.data.DataLoader(self.x_in_bc, sampler=sampler_train)

            else:
                if dist.is_initialized():
                    #our stuff
                    raise NotImplementedError
                else:
                    sampler_train = torch.utils.data.sampler.BatchSampler(
                                    torch.utils.data.sampler.RandomSampler(self.x_interior),
                                    batch_size=int(len(self.x_interior)),
                                    drop_last=False)

                    self.train_loader = torch.utils.data.DataLoader(self.x_interior, sampler=sampler_train)


            print("len(self.x_interior) ", len(self.x_interior))

            # exit(0)


        if(self.ref_strategy=="RAD"):

            c = 0.0
            k = 2.0

            num_samples_train_total = self.num_train_samples_shapes[0]*self.num_train_samples_shapes[1]


            coords_old = copy.deepcopy(self.x_interior)

            self.use_cuda = torch.cuda.is_available()
            self.x_interior, self.xBC_train = self.create_RectangleDomain2D_train(self.num_train_samples_shapes, num_samples_train_total)


            if self.use_cuda:
                self.x_interior = self.x_interior.cuda()
            self.x_interior.requires_grad = True


            # CHECK for BC 
            net_out = net(self.x_interior)

            if(pde.bc_exact is not None):
                net_out = pde.bc_exact.apply_bc(net_out, self.x_interior)


            residual = pde.residual(net_out, self.x_interior)
            residual = torch.abs(residual)

            err_eq   = (torch.pow(residual, k) / torch.pow(residual, k).mean()) + c
            err_eq_normalized = err_eq/ sum(err_eq)


            if self.use_cuda:
                X_ids = np.random.choice(a=err_eq_normalized.shape[0], size=10, replace=False, p=err_eq_normalized.cpu().detach().numpy())
                # X_ids = np.random.choice(a=err_eq_normalized.shape[0], size=num_samples_train_total, replace=False, p=err_eq_normalized)
            else:
                # X_ids = np.random.choice(a=err_eq_normalized.shape[0], size=num_samples_train_total, replace=False, p=err_eq_normalized)
                X_ids = np.random.choice(a=err_eq_normalized.shape[0], size=10, replace=False, p=err_eq_normalized)


            if self.use_cuda:
                self.x_interior = self.x_interior.cpu().detach().numpy()[X_ids]
            else:
                self.x_interior = self.x_interior[X_ids]


            np.set_printoptions(threshold=sys.maxsize)

            
            # print("residual ", residual)
            # print("\n \n err_eq_normalized ", err_eq_normalized)


            self.x_interior = torch.cat((torch.tensor(self.x_interior), coords_old), dim=0)


            # print("\n \n X_ids ", len(self.x_interior))
            # exit(0)


            # train sampler
            if(self.use_BC):
                self.x_in_bc = torch.cat((self.x_interior, self.xBC_train), dim=0)

                if dist.is_initialized():
                    #our stuff
                    raise NotImplementedError
                else:
                    sampler_train = torch.utils.data.sampler.BatchSampler(
                                    torch.utils.data.sampler.RandomSampler(self.x_in_bc),
                                    batch_size=int(len(self.x_in_bc)),
                                    drop_last=False)

                    self.train_loader = torch.utils.data.DataLoader(self.x_in_bc, sampler=sampler_train)

            else:
                if dist.is_initialized():
                    #our stuff
                    raise NotImplementedError
                else:
                    sampler_train = torch.utils.data.sampler.BatchSampler(
                                    torch.utils.data.sampler.RandomSampler(self.x_interior),
                                    batch_size=int(len(self.x_interior)),
                                    drop_last=False)

                    self.train_loader = torch.utils.data.DataLoader(self.x_interior, sampler=sampler_train)


            print("len(self.x_interior) ", len(self.x_interior))



        elif(self.ref_strategy=="EVO"):
            # print(" EVO ")

            self.use_cuda = torch.cuda.is_available()

            for batch_idx, (coordinates) in enumerate(self.train_loader):

                coordinates = coordinates[0]
                coordinates = coordinates.cuda()

                coordinates.requires_grad = True
                            
            
                net_out = net(coordinates)

                if(pde.bc_exact is not None):
                    net_out = pde.bc_exact.apply_bc(net_out, coordinates)


                residual = pde.residual(net_out, coordinates)
                residual = torch.abs(residual)
                
                treshold = torch.mean(residual)
                treshold = treshold.item()

                # print("treshold ", treshold)

                coordinates = coordinates.cpu().detach().numpy()
                coordinates = coordinates[residual.cpu().detach().numpy() > treshold]
                coordinates = torch.tensor(coordinates)



                num_samples_train_total = self.num_train_samples_shapes[0]*self.num_train_samples_shapes[1]
                num_samples_new         = num_samples_train_total - coordinates.shape[0]

  
                self.x_interior, self.xBC_train = self.create_RectangleDomain2D_train(self.num_train_samples_shapes, num_samples_new)


                self.x_interior = torch.cat((self.x_interior, coordinates), dim=0)



            # train sampler
            if(self.use_BC):
                self.x_in_bc = torch.cat((self.x_interior, self.xBC_train), dim=0)

                if dist.is_initialized():
                    #our stuff
                    raise NotImplementedError
                else:
                    sampler_train = torch.utils.data.sampler.BatchSampler(
                                    torch.utils.data.sampler.RandomSampler(self.x_in_bc),
                                    batch_size=int(len(self.x_in_bc)),
                                    drop_last=False)

                    self.train_loader = torch.utils.data.DataLoader(self.x_in_bc, sampler=sampler_train)

            else:
                if dist.is_initialized():
                    #our stuff
                    raise NotImplementedError
                else:
                    sampler_train = torch.utils.data.sampler.BatchSampler(
                                    torch.utils.data.sampler.RandomSampler(self.x_interior),
                                    batch_size=int(len(self.x_interior)),
                                    drop_last=False)

                    self.train_loader = torch.utils.data.DataLoader(self.x_interior, sampler=sampler_train)


        elif(self.ref_strategy=="None"):
            print("No refirement used ")



    def append_analytical_sol_test(self, pde):

        if(pde.has_analytical_sol):
            exact_sol = pde.analytical_u(self.x_test)
            exact_sol = torch.reshape(exact_sol, (self.x_test.shape[0], 1 ))
            new_dataset = torch.hstack((self.x_test, exact_sol))
        else:
            new_dataset = pde.get_test_set()
            sampler_test =  torch.utils.data.sampler.BatchSampler(
                            torch.utils.data.sampler.RandomSampler(new_dataset),
                            batch_size=int(len(new_dataset)),
                            drop_last=False)

        if dist.is_initialized():
            #our stuff
            self.test_loader = GeneralizedDistributedDataLoader(model_structure=self.model_structure, dataset=new_dataset, batch_size=int(len(new_dataset)), shuffle=False, num_workers=0, pin_memory=True)
        else:
            self.test_loader = torch.utils.data.DataLoader(new_dataset, sampler=sampler_test)




    def create_RectangleDomain2D_test(self, num_test_samples):
        # TODO:: add points 
        self.x_test   = self.grid_sampling(num_test_samples, True)
        


    def create_RectangleDomain2D_train(self, num_train_samples, num_samples_total):
        self.vol_domain = torch.tensor(
            [[(self.endPoint_x - self.startPoint_x)*(self.endPoint_y - self.startPoint_y)]])


        if(self.sampling_strategy=="grid"):
            x_interior = self.grid_sampling(num_train_samples, False)
        elif(self.sampling_strategy=="latin_hypercube"):
            x_interior = self.latin_hypercube_sampling(num_train_samples, num_samples_total)
        elif(self.sampling_strategy=="uniform_random"):
            x_interior = self.uniform_random_sampling(num_train_samples, num_samples_total)            
        elif(self.sampling_strategy=="halton"):
            x_interior = self.halton_sampling(num_train_samples, num_samples_total)       
        elif(self.sampling_strategy=="sobol"):
            x_interior = self.sobol_sampling(num_train_samples, num_samples_total)                   
        elif(self.sampling_strategy=="hammersly"):
            x_interior = self.hammersly_sampling(num_train_samples, num_samples_total)                               
        else:
            x_interior = self.latin_hypercube_sampling(num_train_samples, num_samples_total)


        # print("self.x_interior ", self.x_interior.shape)


        # # bc point generation  - uniform grid 
        x_flat = np.linspace(self.startPoint_x, self.endPoint_x,
                             num_train_samples[0], endpoint=True)
        y_flat = np.linspace(self.startPoint_y, self.endPoint_y,
                             num_train_samples[1], endpoint=True)


        x_flat = x_flat.reshape((-1, 1))
        y_flat = y_flat.reshape((-1, 1))

        z_x = np.zeros((num_train_samples[1], 1)) + self.startPoint_x
        o_x = np.ones((num_train_samples[1], 1)) * self.endPoint_x

        z_y = np.zeros((num_train_samples[0], 1)) + self.startPoint_y
        o_y = np.ones((num_train_samples[0], 1)) * self.endPoint_y

        self.leftBC = torch.tensor(np.concatenate((z_x, y_flat), axis=1))
        self.rightBC = torch.tensor(np.concatenate((o_x, y_flat), axis=1))
        self.bottomBC = torch.tensor(np.concatenate((x_flat, z_y), axis=1))
        self.topBC = torch.tensor(np.concatenate((x_flat, o_y), axis=1))

        xBC_train = torch.cat((self.leftBC, self.rightBC), dim=0)
        xBC_train = torch.cat((xBC_train, self.bottomBC), dim=0)
        xBC_train = torch.cat((xBC_train, self.topBC), dim=0)

        return x_interior, xBC_train


    def latin_hypercube_sampling(self, num_samples):
        total_samples = num_samples[0] * num_samples[1]
        x = pyDOE.lhs(2, samples=total_samples)
        x[:, 0] = x[:, 0] * (self.endPoint_x - self.startPoint_x) + self.startPoint_x
        x[:, 1] = x[:, 1] * (self.endPoint_y - self.startPoint_y) + self.startPoint_y

        x = torch.tensor(x)
        return x



    def grid_sampling(self, num_samples, with_BC):

        if with_BC:
    
            x_flat = np.linspace(
                self.startPoint_x, self.endPoint_x, num_samples[0], endpoint=True)
            y_flat = np.linspace(
                self.startPoint_y, self.endPoint_y, num_samples[1], endpoint=True)
            x_flat = x_flat.reshape((-1, 1))
            y_flat = y_flat.reshape((-1, 1))


            X, Y = np.meshgrid(x_flat, y_flat)
            x = np.stack([X.flatten(), Y.flatten()], axis=-1)
            x = torch.tensor(x)

        elif not with_BC:

            hx = (self.endPoint_x - self.startPoint_x)/(num_samples[0]-1)
            hy = (self.endPoint_y - self.startPoint_y)/(num_samples[1]-1)


            x_flat = np.linspace(self.startPoint_x+hx, self.endPoint_x, num_samples[0], endpoint=False)
            y_flat = np.linspace(self.startPoint_y+hy, self.endPoint_y, num_samples[1], endpoint=False)


            x_flat = x_flat.reshape((-1, 1))
            y_flat = y_flat.reshape((-1, 1))


            X, Y = np.meshgrid(x_flat, y_flat)
            x = np.stack([X.flatten(), Y.flatten()], axis=-1)
            x = torch.tensor(x)


        return x



    def uniform_random_sampling(self, num_samples, num_samples_total=None):

        x = np.random.uniform([self.startPoint_x, self.startPoint_y], [self.endPoint_x, self.endPoint_y], (num_samples_total, 2))
        return  torch.tensor(x)


    def halton_sampling(self, num_samples, num_samples_total=None):

        space = [(self.startPoint_x, self.endPoint_x), (self.startPoint_y, self.endPoint_y)]
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)

        x = np.array(sampler.generate(space, num_samples_total))

        return  torch.tensor(x)



    def sobol_sampling(self, num_samples, num_samples_total=None):

        space   = [(self.startPoint_x, self.endPoint_x), (self.startPoint_y, self.endPoint_y)]
        sampler = skopt.sampler.Sobol(skip=0, randomize=False)
        x = np.array(sampler.generate(space, num_samples_total))

        return  torch.tensor(x)



    def hammersly_sampling(self, num_samples, num_samples_total=None):

        space   = [(self.startPoint_x, self.endPoint_x), (self.startPoint_y, self.endPoint_y)]
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
        x = np.array(sampler.generate(space, num_samples_total))

        return  torch.tensor(x)


