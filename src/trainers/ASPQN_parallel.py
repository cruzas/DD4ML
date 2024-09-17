import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.distributed as dist
import sys
import numpy as np
import time
import os
import sys
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler_torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from trainers.Lineasearches import _cubic_interpolate, _strong_wolfe

from trainers.Config import *
from trainers.PrecLBFGS import *


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s    


class ASPQN_parallel(object):
    def __init__(self, training_pinn_flg, num_subdomains, type="classification", freq_print=1, use_wandb=False):
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



    # 1 global optim step, to make sure that gradient and other arrays are allocated
    def init_global_optim(self, net, coordinates, criterion, global_optimizer_type,  args): 
        if(self.training_pinn_flg==False):
            inputs, targets= coordinates
        else:
            coordinates = coordinates[0]

        
        if(self.training_pinn_flg):
            # if self.use_cuda:
            #     coordinates = coordinates.cuda()                       
            # coordinates.requires_grad=True
            # loss, __  = criterion(self.global_net, coordinates)      
            
            def closure():
                nonlocal coordinates

                self.global_optimizer.zero_grad()
                if self.use_cuda:
                    coordinates = coordinates.cuda()                       
                coordinates.requires_grad=True
                __, loss, pde_loss  = criterion(self.global_net, coordinates)            
                loss.backward()  # Backward Propagation
                return loss
        else:
            self.global_optimizer.zero_grad()
            if self.use_cuda:
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)                                          
            u_net = self.global_net(inputs)
            loss = criterion(u_net, targets)
            loss.backward()  # Backward Propagation

        if(args.use_SPQN==False):
            # this is here just for testing purposes

            if(global_optimizer_type==PrecLBFGS):
                self.global_optimizer.step(closure) # Optimizer update  
            else:
                loss = closure()             
                self.global_optimizer.step() # Optimizer update  


        parameters = list(self.global_net.parameters())
        

        with torch.no_grad():
            for global_param, local_param in zip(self.global_net.parameters(), net.parameters()):
                local_param.data.copy_(global_param.data, non_blocking=True)

        self.global_opt_initialized=True      



    def train(self, dataset, net, criterion, global_optimizer_type, local_optimizer_type, args,  regularization=None, lr_scheduler_data=None, lr_scheduler_data_local=None):

        if self.use_cuda:
            net.cuda()

        # broadcasting initial parameters to all nodes, so all nets have same configuration
        with torch.no_grad():
            for p in net.parameters():
                dist.broadcast(p, src=0)

        # this should be implemented way better 
        self.global_net = copy.deepcopy(net)


        if self.use_cuda:
            self.global_net.cuda()
            torch.backends.cudnn.benchmark = True



        if(global_optimizer_type==optim.SGD):
            self.global_optimizer = global_optimizer_type(self.global_net.parameters(), lr=args.lr_global, momentum=0.9)
        elif(global_optimizer_type==PrecLBFGS):
            self.global_optimizer = global_optimizer_type(self.global_net.parameters(), history_size=args.history)
        else:    
            self.global_optimizer = global_optimizer_type(self.global_net.parameters(), lr=args.lr_global)



        # self.local_optimizer = local_optimizer_type(net.extract_trainable_params(sbd_id=dist.get_rank(), num_subdomains=dist.get_world_size()), lr=args.lr_local)
        self.local_optimizer = local_optimizer_type(net.extract_trainable_params(sbd_id=dist.get_rank(), num_subdomains=dist.get_world_size(), overlap_width=args.overlap_width))
        self.scale = net.avg
        dist.all_reduce(self.scale, op=dist.ReduceOp.SUM)

        epoch=0
        self.best_loss_test = 9e9
        self.best_loss = 9e9

        # if(self.config["resume_from_checkpoint"]==False):
        #     #net.init_params()
        # else:
        #     checkpoint = torch.load(self.config["checkpoint_folder_name"] + os.sep  + self.config["checkpoint_file_name"]+'.t7')
        #     net = checkpoint['net']
        #     epoch = checkpoint['epoch']+1
        

        self.print_init()
        elapsed_time = 0
        converged=False


        # TODO:: see if there is a better way of doing this
        # if(args.use_SPQN==True):
        #     # lr_scheduler = lr_scheduler_torch.MultiStepLR(self.global_optimizer, milestones=[int(50/args.num_local_steps), int(100/args.num_local_steps), int(150/args.num_local_steps)], gamma=0.1)
        #     lr_scheduler = lr_scheduler_torch.MultiStepLR(self.global_optimizer, milestones=[50, 100, 150], gamma=0.1)
        # else:
        #     lr_scheduler = lr_scheduler_torch.MultiStepLR(self.global_optimizer, milestones=[50, 100, 150], gamma=0.1)

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

            train_loss, pde_loss, train_acc = self.train_step(epoch, dataset, net, criterion, global_optimizer_type, local_optimizer_type, regularization, args)


            loss_test, test_acc, L2_error, L2_error_rel = self.test_step(dataset, self.global_net, criterion, epoch, train_loss, train_acc)
                

            if(epoch==0):
                args.use_SPQN=flg_SPQN
                if(flg_SPQN==True):
                     for g in self.global_optimizer.param_groups:
                         g['lr'] = args.lr_global
            

            epoch_time = time.time() - start_time
            elapsed_time += epoch_time

            self.num_loss_evals                 +=   self.global_optimizer.num_fun_evals
            self.global_optimizer.num_fun_evals = 0.0
            self.num_loss_evals                 +=  self.local_optimizer.num_fun_evals
            self.local_optimizer.num_fun_evals  = 0.0
            self.num_grad_evals                 =   2*epoch + (epoch*args.num_local_steps)            

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
            elif(self.num_grad_evals >= self.config["max_epochs"] or self.num_grad_evals >= args.max_epochs):
                converged=True                
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


            # if((epoch-1)%self.freq_print==0 and dist.get_rank()==0):
            #     wandb.log({'epoch': epoch, 'loss': loss})


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

        loss_global, flat_grad, lr, ls_func_evals = _strong_wolfe(
            obj_func, x_global_init, 1.0, correction, loss_global_old, flat_grad_old, gtd)                    

        self.global_optimizer._add_dir(lr, correction)

        return loss_global, flat_grad


    def get_loss_grad(self, coordinates, criterion): 
        self.global_optimizer.zero_grad()

        if self.use_cuda:
            coordinates = coordinates.cuda()                       
        coordinates.requires_grad=True

        __, loss, __  = criterion(self.global_net, coordinates)
        loss.backward()

        grad       = self.global_optimizer._gather_flat_grad() 

        return loss, grad      


    @torch.no_grad()
    def collect_local_corrections(self, net): 
        corr = [p_local - p for p, p_local in zip(self.global_net.parameters(), net.parameters())]
        correction = self.global_optimizer._gather_flat(corr)
        dist.all_reduce(correction, op=dist.ReduceOp.SUM)
        return correction / self.scale


    def train_step(self, epoch, dataset, net, criterion, global_optimizer_type, local_optimizer_type, regularization, args):
        net.train()
        net.training = True
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
                self.init_global_optim(net, coordinates, criterion, global_optimizer_type, args)            
                
                if(args.use_SPQN==True):
                    break



        if(args.use_SPQN==True):

            dataloader_iterator = iter(dataset.train_loader)
            while(stop_batching==False):
                start_time = time.time()
                try:
                    if(self.training_pinn_flg==False):
                        inputs, targets = next(dataloader_iterator)
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

                # print("Getting data ", dist.get_rank(),  "synch step " , time.time() - start_time)
                
                # solve on subdomains
                start_time = time.time()


                for local_it in range(0, args.num_local_steps):

                    def closure_local():
                        nonlocal coordinates
                        self.local_optimizer.zero_grad()
                        
                        if(self.training_pinn_flg):
                            if self.use_cuda:
                                coordinates = coordinates.cuda()  
                            coordinates.requires_grad=True                
                            __, loss_loc, pde_loss_loc = criterion(net, coordinates)
                        else:
                            if self.use_cuda:
                                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)                      
                            u_net_loc = net(inputs)
                            loss_loc = criterion(u_net_loc, targets)

                        
                        if(regularization is not None):
                            reg_loss_loc = regularization(net)
                            loss_loc += reg_loss_loc


                        start_time = time.time()
                        loss_loc.backward()  # Backward Propagation


                        # if(dist.get_rank()==0):
                        #     print("loss_loc   ", loss_loc.item())

                        return loss_loc

                    loss_loc = self.local_optimizer.step(closure_local) # Optimizer update

                # # # # # # # # # # # # # # # get correction from local to global  # # # # # # # # # # # # # # # # # 
                loss_global_old, flat_grad_old  = self.get_loss_grad(coordinates, criterion)
                correction = self.collect_local_corrections(net)


                def closure_global():
                    nonlocal coordinates

                    self.global_optimizer.zero_grad()

                    if self.use_cuda:
                        coordinates = coordinates.cuda()                       
                    coordinates.requires_grad=True

                    __, loss, __  = criterion(self.global_net, coordinates)
                    loss.backward()

                    return loss


                loss_global_new, flat_grad_new = self.compute_lr_local_to_global(closure_global, correction, flat_grad_old, loss_global_old)

                
                # # update LBFGS old history 
                # global_optim_state                      = self.global_optimizer.state[self.global_optimizer._params[0]]
                # flat_grad_new                           = self.global_optimizer._gather_flat_grad()
                # global_optim_state                   = self.global_optimizer.state[self.global_optimizer._params[0]]
                # global_optim_state['prev_flat_grad'] = flat_grad_new #flat_grad_new, flat_grad


                
                # # # # # # # # # # # # # # #  Now, take LBFGS step for x^+  # # # # # # # # # # # # # # # # #                 
                if(global_optimizer_type==PrecLBFGS):
                    self.global_optimizer.step(closure_global) # Optimizer update
                else:
                    self.global_optimizer.step() 




                # print("Global step ", dist.get_rank(),  "synch step " , time.time() - start_time)



                # # # # # # # # # # # # # # # Synch global net to local (all cores)  # # # # # # # # # # # # # # #
                start_time = time.time()
                # synchronize local nets with global one 
                with torch.no_grad():
                    for global_param, local_param in zip(self.global_net.parameters(), net.parameters()):
                        local_param.data.copy_(global_param.data, non_blocking=True)


                # print("Copy local to global step ", dist.get_rank(),  "synch step " , time.time() - start_time)
                # exit(0)


         
        # this loop can be paralelized over samples 
        if(epoch%self.freq_print==0):
        # if(epoch%1==0):
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
        if(dist.get_rank()==0):
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Epoch.     #g_ev        #l_ev      L_inner(train)        L_sum(train)          L_sum(test)           L2_error            L2_error_rel          time ")
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")



    def print_epoch(self, epoch, pde_loss,  train_loss, test_loss, L2_error, L2_error_rel, time_current):
        if(epoch % self.freq_print == 0):
            if(dist.get_rank()==0):
                print("{:03d}         {:03d}         {:03d}         {:e}         {:e}         {:e}         {:e}         {:e}        {:s} ".format(
                    epoch, int(self.num_grad_evals), int(self.num_loss_evals), pde_loss, train_loss, test_loss, L2_error, L2_error_rel, time_current))

            if(dist.get_rank()==0):
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
