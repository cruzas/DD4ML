import torch
import torch.optim as optim
import numpy as np
import time
import os
import pandas as pd
from trainers.Lineasearches import _strong_wolfe

from trainers.Config import *
from trainers.PrecLBFGS import *

from abc import abstractmethod

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s    

class SPQNTrainerRight(object):
    def __init__(self, training_pinn_flg, num_subdomains, type="regression", freq_print=1, use_wandb=False, use_coarse=False, layerwise_lr_update=False, stochastic_update=False):
        self.config = copy.deepcopy(trainer_config)
        self.use_cuda = torch.cuda.is_available()        
        self.training_pinn_flg = training_pinn_flg
        self.num_subdomains = num_subdomains
        self.type = type

        self.best_acc_train = 0.0
        self.best_acc_test = 0.0

        self.global_opt_initialized=False 

        self.freq_print = freq_print
        self.use_wandb = use_wandb

        self.red_strategy = 'flatten_local' #"flatten_all" # "all" # 'flatten_local'


        self.convergence_status = dict( epochs=0,
                                        num_loss_evals=0,
                                        num_grad_evals=0,
                                        loss_train=0, 
                                        loss_val=0, 
                                        L2_error=0, 
                                        L2_error_rel=0)

        self.num_grad_evals = 0.0
        self.num_loss_evals = 0.0
        
        self.L2_error       = 9e9
        self.L2_error_rel   = 9e9           

        # variable for twolevel
        self.use_coarse = use_coarse

        self.layerwise_lr_update = layerwise_lr_update
        self.stochastic_update = stochastic_update



    def save_model(self, net, name):
        self.create_folder('models')
        save_path = os.path.join('models', name)

        # if dist.is_initialized() and dist.get_world_size()>1:
        #     torch.save(net.module.state_dict(), save_path)
        # else:
        torch.save(net.state_dict(), save_path)


    
    def create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            try:
                os.mkdir(folder_path)
            except OSError as error:
                pass



    def load_model(self, net, args):
        if(args.model_name_load=="none"):
            PATH = os.path.join('models', args.model_name)
            print("load_model:: PATH ", PATH)
        else:
            PATH = os.path.join('models', args.model_name_load)
            print("load_model:: PATH ", PATH)

        if(os.path.isfile(PATH)):

            if torch.cuda.is_available():
                DEVICE = torch.device("cuda:0")
            else:
                DEVICE = torch.device("cpu")
            
            net.load_state_dict(torch.load(PATH, map_location=DEVICE))
            return True
        else:
            print("DeepOnet not found ... ")
            return False
            



    # 1 global optim step, to make sure that gradient and other arrays are allocated
    def init_global_optim(self, coordinates, criterion, global_optimizer_type,  args): 
        if(self.training_pinn_flg==False):
            inputs, targets= coordinates
            
            def closure():
                nonlocal inputs
                nonlocal targets

                self.global_optimizer.zero_grad()
                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()   

                u_net = self.global_net(inputs)
                loss  = criterion(u_net, targets)            

                loss.backward()  # Backward Propagation
                return loss            
        else:
            coordinates = coordinates[0]

            def closure():
                nonlocal coordinates

                self.global_optimizer.zero_grad()
                if self.use_cuda:
                    coordinates = coordinates.cuda()                       
                coordinates.requires_grad=True
                __, loss, pde_loss  = criterion(self.global_net, coordinates)            
                loss.backward()  # Backward Propagation
                return loss

        if(args.use_SPQN==False):
            # this is here just for testing purposes

            if(global_optimizer_type==PrecLBFGS):
                self.global_optimizer.step(closure) # Optimizer update  
            else:
                loss = closure()             
                self.global_optimizer.step() # Optimizer update  

        self.global_opt_initialized=True      

    def define_global_network(self, net, global_optimizer_type, args):
        # global network
        global_net = copy.deepcopy(net)

        if(global_optimizer_type==optim.SGD):
            global_optimizer = global_optimizer_type(global_net.parameters(), lr=args.lr_global, momentum=0.9)
        elif(global_optimizer_type==PrecLBFGS):
            global_optimizer = global_optimizer_type(global_net.parameters(), history_size=args.history)
        else:    
            global_optimizer = global_optimizer_type(global_net.parameters(), lr=args.lr_global)
        return global_net, global_optimizer


    def define_coarse_network(self, net, global_optimizer_type, args):
        # global network
        coarse_net = copy.deepcopy(net)
        
        if(global_optimizer_type==PrecLBFGS):
            # coarse_optimizer = global_optimizer_type(coarse_net.parameters(), history_size=args.history)
            num_coarse_sbds = args.num_subdomains

            # print("\n \n Coarse net decomp")
            # coarse_net.print_decomposition(num_coarse_sbds)
            coarse_optimizer = global_optimizer_type(coarse_net.extract_coarse_trainable_params(num_subdomains=num_coarse_sbds, overlap_width=0))
            

        else:    
            # coarse_optimizer = global_optimizer_type(coarse_net.parameters(), lr=args.lr_global)
            print("TODO:: add coarse nets  ")
            exit(0)
        

        return coarse_net, coarse_optimizer




    def define_local_networks(self, net, local_optimizer_type, args):
        # local networks
        local_nets = [copy.deepcopy(net) for _ in range(self.num_subdomains)]
        local_optimizers = [local_optimizer_type(local_nets[sbd_id].extract_trainable_params(sbd_id=sbd_id, num_subdomains=args.num_subdomains, overlap_width=args.overlap_width)) for sbd_id in range(self.num_subdomains)]
        self.scale = sum([loc_net.get_avg() for loc_net in local_nets])
        self.coloring = torch.mean(self.scale)

        # print("self.coloring ", self.coloring)
        # exit(0)

        return local_nets, local_optimizers


    @abstractmethod
    def compute_num_loss_grad_evals(self, num_loss_evals, epoch, args):
        raise NotImplementedError


    @abstractmethod
    def local_to_global_correction(self, closure_global, local_nets, coordinates, criterion):         
        raise NotImplementedError


    def train(self, dataset, net, criterion, global_optimizer_type, local_optimizer_type, args,  regularization=None, lr_scheduler_data=None, lr_scheduler_data_local=None):
        if self.use_cuda:
            net.cuda()
            torch.backends.cudnn.benchmark = True


        # global network
        self.global_net, self.global_optimizer = self.define_global_network(net, global_optimizer_type, args)

        # local network
        self.local_nets, self.local_optimizers = self.define_local_networks(net, local_optimizer_type, args)
        
        # coarse network
        self.coarse_net, self.coarse_optimizer = self.define_coarse_network(net, global_optimizer_type, args)


        epoch=0
        self.best_loss_test = 9e9
        self.best_loss = 9e9


        self.print_init()
        elapsed_time = 0
        converged=False

        if(lr_scheduler_data is not None):
            sched_type, milestones_user, gamma_user = lr_scheduler_data
            lr_scheduler = sched_type(self.global_optimizer, milestones_user, gamma_user)
        else:
            lr_scheduler = None

        if(lr_scheduler_data_local is not None):
            sched_type, milestones_user, gamma_user = lr_scheduler_data_local
            lr_scheduler_local = sched_type(self.local_optimizer, milestones_user, gamma_user)
        else:
            lr_scheduler_local = None



        while(converged==False):
            start_time = time.time()

            if(epoch==0):
                flg_SPQN = args.use_SPQN
                args.use_SPQN=False


            train_loss, pde_loss, train_acc = self.train_step(epoch, dataset, criterion, global_optimizer_type, local_optimizer_type, regularization, args)
            loss_test, test_acc, L2_error, L2_error_rel = self.test_step(dataset, self.global_net, criterion, epoch, train_loss, train_acc)
                

            if(epoch==0):
                args.use_SPQN=flg_SPQN
                if(flg_SPQN==True):
                     for g in self.global_optimizer.param_groups:
                         g['lr'] = args.lr_global
            

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time

            # compute num of loss, grad evals
            self.num_loss_evals, self.num_grad_evals = self.compute_num_loss_grad_evals(self.num_loss_evals, epoch, args)

            epoch+=1

            time_current = str('%d:%02d:%02d'  %(get_hms(elapsed_time)))
            # self.print_epoch(epoch, pde_loss, loss_with_reg, loss_test, train_acc, test_acc,  time_current)
            self.print_epoch(epoch, pde_loss, train_loss, loss_test, L2_error, L2_error_rel,  time_current)
            

            if(lr_scheduler is not None):
                 lr_scheduler.step()


            if(lr_scheduler_data_local is not None):
                # for iii in range(0, args.num_local_steps):
                lr_scheduler_local.step()


            if(epoch >= self.config["max_epochs"] or epoch >= args.max_epochs):
                converged=True
            # elif(self.num_grad_evals >= self.config["max_epochs"] or self.num_grad_evals >= args.max_epochs):
            #     converged=True                        
            elif(train_loss < self.config["loss_train_tol"] or loss_test < self.config["loss_test_tol"]):
                converged=True
            elif(np.isfinite(train_loss)==False or np.isfinite(pde_loss)==False or np.isfinite(loss_test)==False): 
                converged=True                

            if (self.type == "classification"):
                if(train_acc >= self.config["acc_train_tol"] and self.type == "classification"):
                    converged=True
                elif(test_acc >= self.config["acc_val_tol"] and self.type == "classification"):
                    converged=True                
                elif(epoch > 3 and train_acc < 15):
                    converged=True

        if(args.use_SPQN==False):
            with torch.no_grad():
                for global_param, local_param in zip(self.global_net.parameters(), net.parameters()):
                    local_param.data.copy_(global_param.data, non_blocking=True)

        self.convergence_status["epochs"]           = epoch
        self.convergence_status["num_loss_evals"]   = self.num_loss_evals
        self.convergence_status["num_grad_evals"]   = self.num_grad_evals

        self.convergence_status["loss_train"]   = self.best_loss
        self.convergence_status["loss_val"]     = self.best_loss_test
        
        self.convergence_status["L2_error"]     = self.L2_error
        self.convergence_status["L2_error_rel"] = self.L2_error_rel

        return self.convergence_status

    @torch.no_grad()
    def compute_lr_local_to_global(self, closure_global, correction, flat_grad_old, loss_global_old):

        closure_global = torch.enable_grad()(closure_global)

        def obj_func(x, t, d):
            return self.global_optimizer._directional_evaluate(closure_global, x, t, d)

        gtd = flat_grad_old.dot(correction)

        x_global_init = self.global_optimizer._clone_param()

        loss_global, flat_grad, lr, ls_func_evals = _strong_wolfe(obj_func, x_global_init, 1.0, correction, loss_global_old, flat_grad_old, gtd)

        self.global_optimizer._add_dir(lr, correction)

        return loss_global, flat_grad

    @abstractmethod
    def local_corrections(self, closure_global, coordinates, criterion):
        raise NotImplementedError

    def get_loss_grad(self, coordinates, criterion): 
        self.global_optimizer.zero_grad()

        if(self.training_pinn_flg == True):
            if self.use_cuda:
                coordinates = coordinates.cuda()  
            coordinates.requires_grad=True                
            __, loss, __ = criterion(self.global_net, coordinates)
        else:
            inputs, targets = coordinates
            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            u_net = self.global_net(inputs)
            loss = criterion(u_net, targets)

        loss.backward()
        grad       = self.global_optimizer._gather_flat_grad() 

        return loss, grad



    @abstractmethod
    def transfer_update(self, sbd_id):
        raise NotImplementedError

    def train_step(self, epoch, dataset, criterion, global_optimizer_type, local_optimizer_type, regularization, args):
        loss_no_reg = 0
        loss_with_reg = 0
        loss=0
        total = 0
        state_global_step=0
        train_acc=0
        correct = 0

        stop_batching=False

        if(self.global_opt_initialized==False or args.use_SPQN==False): 
            for batch_idx, (coordinates) in enumerate(dataset.train_loader):
                self.init_global_optim(coordinates, criterion, global_optimizer_type, args)            
                if(args.use_SPQN==True):
                    break


        if(args.use_SPQN==True):

            dataloader_iterator = iter(dataset.train_loader)
            while(stop_batching==False):
                start_time = time.time()
                try:
                    if(self.training_pinn_flg==False):
                        coordinates = next(dataloader_iterator)
                    else:
                        coordinates     = next(dataloader_iterator)
                        coordinates     = coordinates[0]

                except StopIteration:
                    dataloader_iterator = iter(dataset.train_loader)
                    if(self.training_pinn_flg==False):
                        inputs, targets = next(dataloader_iterator)
                    else:
                        coordinates     = next(dataloader_iterator)   
                        coordinates     = coordinates[0]

                    stop_batching = True   
                    break           
                
                # solve on subdomains
                start_time = time.time()


                if(self.use_coarse):

                    # DOING some SORT of COARSE step 
                    # self.coarse_net, self.coarse_optimizer                
                    for param_global, param_coarse in zip(self.global_net.parameters(), self.coarse_net.parameters()):
                        param_coarse.data = param_coarse.data.copy_(param_global.data)                


                    for coarse_it in range(0, args.num_coarse_steps):
                        def closure_coarse():
                            nonlocal coordinates
                            self.coarse_optimizer.zero_grad()
                            

                            if(self.training_pinn_flg == True):
                                if self.use_cuda:
                                    coordinates = coordinates.cuda()  
                                coordinates.requires_grad=True                
                                __, loss_coarse, pde_loss_loc = criterion(self.coarse_net, coordinates)
                            else:
                                inputs, targets = coordinates
                                if self.use_cuda:
                                    inputs = inputs.cuda()
                                    targets = targets.cuda()

                                u_net = self.coarse_net(inputs)
                                loss_coarse = criterion(u_net, targets)

                            if(regularization is not None):
                                reg_loss_loc = regularization(self.coarse_net)
                                loss_coarse += reg_loss_loc
                            
                            start_time = time.time()
                            loss_coarse.backward()  # Backward Propagation

                            return loss_coarse

                        loss_coarse = self.coarse_optimizer.step(closure_coarse) # Optimizer update



                    # additive
                    def compute_coarse_corrections():
                        corr = [p_coarse - p for p, p_coarse in zip(self.global_net.parameters(), self.coarse_net.parameters())]
                        corr = self.global_optimizer._gather_flat(corr)
                        return corr

                    
                    # global closure 
                    def closure_global():
                        nonlocal coordinates

                        self.global_optimizer.zero_grad()

                        if(self.training_pinn_flg == True):
                            if self.use_cuda:
                                coordinates = coordinates.cuda()  
                            coordinates.requires_grad=True                
                            __, loss, __ = criterion(self.global_net, coordinates)
                        else:
                            inputs, targets = coordinates
                            if self.use_cuda:
                                inputs = inputs.cuda()
                                targets = targets.cuda()

                            u_net = self.global_net(inputs)
                            loss = criterion(u_net, targets)

                        loss.backward()

                        return loss


                    correction = compute_coarse_corrections()
                    loss_global_old, flat_grad_old  = self.get_loss_grad(coordinates, criterion)
                    loss_global_new, flat_grad_new  = self.compute_lr_local_to_global(closure_global, correction, flat_grad_old, loss_global_old)

                

                # USUAL
                # copy global to 1st. local     
                for param_global, param_local in zip(self.global_net.parameters(), self.local_nets[0].parameters()):
                    param_local.data = param_local.data.copy_(param_global.data)


                for sbd_id in range(0, self.num_subdomains):
                    for local_it in range(0, args.num_local_steps):

                        def closure_local():
                            nonlocal coordinates
                            self.local_optimizers[sbd_id].zero_grad()
                            
                            if(self.training_pinn_flg):
                                if self.use_cuda:
                                    coordinates = coordinates.cuda()  
                                coordinates.requires_grad=True                
                                __, loss_loc, pde_loss_loc = criterion(self.local_nets[sbd_id], coordinates)
                            else:
                                inputs, targets = coordinates
                                if self.use_cuda:
                                    inputs = inputs.cuda()
                                    targets = targets.cuda()
                                
                                u_net = self.local_nets[sbd_id](inputs)
                                loss_loc = criterion(u_net, targets)



                            if(regularization is not None):
                                reg_loss_loc = regularization(self.local_nets[sbd_id])
                                loss_loc += reg_loss_loc


                            loss_loc.backward()  # Backward Propagation


                            return loss_loc

                        loss_loc = self.local_optimizers[sbd_id].step(closure_local) # Optimizer update
                    self.transfer_update(sbd_id)



                
                # # # # # # # # # # # # # # # get correction from local to global  # # # # # # # # # # # # # # # # # 
                def closure_global():
                    nonlocal coordinates

                    self.global_optimizer.zero_grad()

                    if(self.training_pinn_flg):
                        if self.use_cuda:
                            coordinates = coordinates.cuda()                       
                        coordinates.requires_grad=True
                        __, loss, __  = criterion(self.global_net, coordinates)
                    else:
                        inputs, targets = coordinates
                        if self.use_cuda:
                            inputs = inputs.cuda()
                            targets = targets.cuda()

                        u_net = self.global_net(inputs)
                        loss = criterion(u_net, targets)

                    loss.backward()

                    return loss


                self.local_to_global_correction(closure_global, self.local_nets, coordinates, criterion)


                # # # # # # # # # # # # # # #  Now, take LBFGS step for x^+  # # # # # # # # # # # # # # # # #                 

                ## TODO:: verify that correction is correct 
                # do stochastic update
                if epoch%self.config['frequency'] == 0 or not self.stochastic_update:
                    if(global_optimizer_type==PrecLBFGS):
                        self.global_optimizer.step(closure_global) # Optimizer update
                    else:
                        self.global_optimizer.step()
         
        # this loop can be paralelized over samples 
        if(epoch%self.freq_print==0):

            self.save_model(self.global_net, args.model_name)

        # if(epoch%1==0):
            pde_loss = torch.tensor(0.0)
            for batch_idx, (coordinates) in enumerate(dataset.train_loader):
                # TODO:: figure out how to have global stats 
                # to have global statistics   
                if(self.training_pinn_flg):
                    coordinates = coordinates[0]
                    if self.use_cuda:
                        coordinates = coordinates.cuda()                       
                    coordinates.requires_grad=True
                    # u_net = self.global_net(coordinates.double())  
                    # loss = criterion(u_net, coordinates)
                    __, loss, pde_loss  = criterion(self.global_net, coordinates)

                else:
                    inputs, targets = coordinates

                    if self.use_cuda:
                        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)                                          
                    u_net = self.global_net(inputs)
                    loss = criterion(u_net, targets)

                loss_no_reg += loss.item()

                if(regularization is not None):
                  reg_loss = regularization(self.global_net)
                  loss += reg_loss
                  # loss_with_reg += loss.item()


                if(self.type == "classification"):
                    with torch.no_grad():
                        _, predicted = torch.max(u_net.data, 1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.data).sum()
                        # print("----TRAIN:  batch_idx ", batch_idx,   " total ", total, "  correct ", correct)


            if(self.type == "classification"):
                train_acc = 100.*correct/total
                # print("xxxxx TRAIN:  train_acc ", train_acc)
  

        # print("loss, ", loss)
        # print("loss_no_reg, ", loss_no_reg)
        # print("loss_with_reg, ", loss_with_reg)

        return loss.item(), pde_loss.item(), train_acc



    def test_step(self, dataset, net, criterion, epoch, loss_with_reg, acc_train):
        
        net.eval()
        net.training = False
        loss_test = 0
        test_acc = 0 
        total = 0
        correct = 0
        l2_error = 0
        l2_error_rel = 0

    
        # with torch.no_grad():
        for batch_idx, (coordinates) in enumerate(dataset.test_loader):

            if(self.training_pinn_flg==False):
                inputs, targets= coordinates
            else:
                coordinates_sol = coordinates[0]

                coordinates = coordinates_sol[:, 0:-1]
                exact_sol   = coordinates_sol[:, -1]

            if(self.training_pinn_flg):
                if self.use_cuda:
                    coordinates = coordinates.cuda()          
                    exact_sol   = exact_sol.cuda()


                coordinates.requires_grad=True
                # u_net = net(coordinates.double())  
                # loss = criterion(u_net, coordinates)
                

                u_net, loss, loss_inner = criterion(net, coordinates)

                l2_error        += torch.norm(exact_sol - u_net).item()
                l2_error_rel    += (l2_error/torch.norm(exact_sol)).item()                


            else:
                if self.use_cuda:
                    inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)                                          
                u_net = net(inputs)
                loss = criterion(u_net, targets)

                l2_error        += torch.norm(targets - u_net).item()
                l2_error_rel    += (l2_error/torch.norm(targets)).item()                
            loss_test += loss.item()

            if(self.type == "classification"):
                _, predicted = torch.max(u_net.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).sum()            


        # if loss_test < self.best_loss_test:
        if loss_with_reg < self.best_loss:
            self.best_loss_test = loss_test
            self.best_loss = loss_with_reg

        if(self.type == "classification"):
            test_acc = 100.*correct/total


        if test_acc < self.best_acc_test:
            self.best_acc_test  = test_acc
            self.best_acc       = acc_train
          

        if l2_error_rel < self.L2_error_rel:
            self.L2_error     = l2_error
            self.L2_error_rel = l2_error_rel     


        if(self.config["checkpoint_folder_name"] is not None):
            if loss_test < self.best_loss_test:
                print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(loss_test))
                state = {
                        'net':net.module if self.use_cuda else net,
                        'loss':loss_test,
                        'epoch':epoch,
                }

                if not os.path.isdir(self.config["checkpoint_folder_name"]):
                    os.mkdir(self.config["checkpoint_folder_name"])
                torch.save(state, self.config["checkpoint_folder_name"] + os.sep  + self.config["checkpoint_file_name"]+'.t7')
                self.best_loss_test = loss_test


        return loss_test, test_acc, l2_error, l2_error_rel

    def print_init(self):
        print("----------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Epoch.     #g_ev        #l_ev      L_inner(train)        L_sum(train)          L_sum(test)           L2_error            L2_error_rel          time ")
        print("----------------------------------------------------------------------------------------------------------------------------------------------------")

    def print_epoch(self, epoch, pde_loss,  train_loss, test_loss, L2_error, L2_error_rel, time_current):
        if(epoch % self.freq_print == 0):
            print("{:03d}         {:03d}         {:03d}         {:e}         {:e}         {:e}         {:e}         {:e}        {:s} ".format(
                epoch, int(self.num_grad_evals), int(self.num_loss_evals), pde_loss, train_loss, test_loss, L2_error, L2_error_rel, time_current))


            if(self.config.get("conv_history_csv_name") is not None):
                df = pd.DataFrame({ 'epoch': [epoch],
                                    'g_evals': [self.num_grad_evals],
                                    'loss_evals': [self.num_loss_evals],
                                    'pde_loss': [pde_loss],
                                    'loss_train': [train_loss],
                                    'loss_val': [test_loss], 
                                    'L2_error': [L2_error],
                                    'L2_error_rel': [L2_error_rel]})

                if not os.path.isfile(self.config.get("conv_history_csv_name")+'.csv'):
                    df.to_csv(self.config.get("conv_history_csv_name") +
                              '.csv', index=False,  header=True, mode='a')
                else:
                    df.to_csv(self.config.get("conv_history_csv_name") +
                              '.csv', index=False, header=False, mode='a')



class ASPQNTrainerRight(SPQNTrainerRight):
    def __init__(self, training_pinn_flg, num_subdomains, type="classification", freq_print=1, use_wandb=False, use_coarse=False, layerwise_lr_update=False, stochastic_update=False):
        super().__init__(training_pinn_flg, num_subdomains, type, freq_print, use_wandb, use_coarse, layerwise_lr_update, stochastic_update)

    @torch.no_grad()
    def compute_num_loss_grad_evals(self, num_loss_evals, epoch, args):

        if(args.use_SPQN):
            # additive
            num_loss_evals                 +=  self.global_optimizer.num_fun_evals
            self.global_optimizer.num_fun_evals = 0.0
            num_loss_evals                 +=  self.local_optimizers[0].num_fun_evals
            
            for loc_opt in self.local_optimizers:
                loc_opt.num_fun_evals   = 0.0

            # TODO:: change based on freq
            if(self.stochastic_update):
                num_grad_evals  = (2 * (epoch/args.freq)) + (epoch * args.num_local_steps)
            else:
                num_grad_evals  = (2 * epoch) + (epoch * args.num_local_steps)

            # scaling works as 1 CP per subdomain
            if self.use_coarse:
                num_loss_evals += self.coarse_optimizer.num_fun_evals* (self.num_subdomains/self.coarse_net.get_num_layers())
                self.coarse_optimizer.num_fun_evals = 0.0
                num_grad_evals += epoch * args.num_coarse_steps*(self.num_subdomains/self.coarse_net.get_num_layers())
        else:
            num_loss_evals +=  self.global_optimizer.num_fun_evals
            self.global_optimizer.num_fun_evals = 0.0
            num_grad_evals  = epoch


        return num_loss_evals, num_grad_evals



    @torch.no_grad()
    def transfer_update(self, sbd_id):
        # additive
        if(sbd_id < self.num_subdomains-1):
            for param_global, param_local in zip(self.global_net.parameters(), self.local_nets[sbd_id+1].parameters()):
                param_local.data = param_local.data.copy_(param_global.data)

    @torch.no_grad()
    def collect_local_corrections(self, local_nets): 
        # additive
        def compute_local_corrections(net):
            corr = [p_local - p for p, p_local in zip(self.global_net.parameters(), net.parameters())]
            corr = self.global_optimizer._gather_flat(corr)
            return corr

        correction = [compute_local_corrections(loc_net) for loc_net in local_nets]
        correction = sum(correction) / self.coloring
        return correction

    def local_to_global_correction(self, closure_global, local_nets, coordinates, criterion): 

        if(self.layerwise_lr_update):
            # additive
            @torch.no_grad()
            def compute_local_corrections(net):
                corr = [p_local - p for p, p_local in zip(self.global_net.parameters(), net.parameters())]
                corr = self.global_optimizer._gather_flat(corr)
                return corr

            correction = [compute_local_corrections(loc_net) for loc_net in local_nets]

            loss_global, flat_grad  = self.get_loss_grad(coordinates, criterion)
            for sbd_id in range(self.num_subdomains):
                loss_global, flat_grad = self.compute_lr_local_to_global(closure_global, correction[sbd_id], flat_grad, loss_global)

        else:
            loss_global_old, flat_grad_old  = self.get_loss_grad(coordinates, criterion)
            correction = self.collect_local_corrections(self.local_nets)
            loss_global_new, flat_grad_new = self.compute_lr_local_to_global(closure_global, correction, flat_grad_old, loss_global_old)


class MSPQNTrainerRight(SPQNTrainerRight):
    def __init__(self, training_pinn_flg, num_subdomains, type="classification", freq_print=1, use_wandb=False, use_coarse=False, stochastic_update=False):
        super().__init__(training_pinn_flg, num_subdomains, type, freq_print, use_wandb, use_coarse, False, stochastic_update)


    @torch.no_grad()
    def compute_num_loss_grad_evals(self, num_loss_evals, epoch, args):
        

        if(args.use_SPQN):
            num_loss_evals                 +=  self.global_optimizer.num_fun_evals
            self.global_optimizer.num_fun_evals = 0.0
            for loc_opt in self.local_optimizers:
                num_loss_evals     += loc_opt.num_fun_evals
                loc_opt.num_fun_evals   = 0.0


            if(self.stochastic_update):
                num_grad_evals  = (2 * (epoch/args.freq)) + (self.num_subdomains * epoch * args.num_local_steps)
            else:
                num_grad_evals  = (2 * epoch) + (self.num_subdomains *epoch * args.num_local_steps)

            # scaling works as 1 CP per subdomain
            if self.use_coarse:
                num_loss_evals += self.coarse_optimizer.num_fun_evals* (self.num_subdomains/self.coarse_net.get_num_layers())
                self.coarse_optimizer.num_fun_evals = 0.0
                num_grad_evals += epoch * args.num_coarse_steps*(self.num_subdomains/self.coarse_net.get_num_layers())

        else:
            num_loss_evals +=  self.global_optimizer.num_fun_evals
            self.global_optimizer.num_fun_evals = 0.0
            num_grad_evals  = epoch            

        return num_loss_evals, num_grad_evals

    @torch.no_grad()
    def transfer_update(self, sbd_id):
        # multiplicative
        if(sbd_id < self.num_subdomains-1):
            for param_local_next, param_local in zip(self.local_nets[sbd_id+1].parameters(), self.local_nets[sbd_id].parameters()):
                param_local_next.data = param_local_next.data.copy_(param_local.data)


    @torch.no_grad()
    def collect_local_corrections(self, local_nets): 
        # multiplicative
        corr = [p_local - p for p, p_local in zip(self.global_net.parameters(), local_nets[-1].parameters())]
        correction = self.global_optimizer._gather_flat(corr)
        return correction


    
    def local_to_global_correction(self, closure_global, local_nets, coordinates, criterion): 

        loss_global_old, flat_grad_old  = self.get_loss_grad(coordinates, criterion)
        correction = self.collect_local_corrections(self.local_nets)

        # line-search should not be necessary, as all updates in local LBFGS were obtained sequentially and by using line-search 
        # loss_global_new, flat_grad_new = self.compute_lr_local_to_global(closure_global, correction, flat_grad_old, loss_global_old)

        @torch.no_grad()
        def add_corr(correction):
            self.global_optimizer._add_dir(1.0, correction)

        
        add_corr(correction)
        




