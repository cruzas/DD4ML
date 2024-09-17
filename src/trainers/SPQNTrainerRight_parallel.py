import torch
import torch.optim as optim
import numpy as np
import time
import os
import pandas as pd
from trainers.Lineasearches import _strong_wolfe

from trainers.Config import *
from trainers.PrecLBFGS import *

import torch.distributed as dist

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s    

class ASPQNTrainerRightDistrib(object):
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


    # 1 global optim step, to make sure that gradient and other arrays are allocated
    def init_global_optim(self, net, dataset, criterion, global_optimizer_type,  args): 

        if(self.training_pinn_flg==False):            
            def closure():
                accum_loss = 0
                self.global_optimizer.zero_grad()

                for batch_idx, (coordinates) in enumerate(dataset.train_loader):
                    inputs, targets= coordinates
                    if self.use_cuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()


                    u_net = self.global_net(inputs)
                    loss  = criterion(u_net, targets)/args.num_batches

                    loss.backward()  # Backward Propagation
                    accum_loss += loss

                return accum_loss

        else:
            coordinates = coordinates[0]

            def closure():

                accum_loss = 0
                self.global_optimizer.zero_grad()

                for batch_idx, (coordinates) in enumerate(dataset.train_loader):
                    if self.use_cuda:
                        coordinates = coordinates.cuda()
                    coordinates.requires_grad=True
                    __, loss, pde_loss  = criterion(self.global_net, coordinates)

                    loss=loss/args.num_batches
                    loss.backward()  # Backward Propagation
                    accum_loss += loss
                
                return accum_loss



        if(args.use_SPQN==False):
            # this is here just for testing purposes

            if(global_optimizer_type==PrecLBFGS):
                loss = self.global_optimizer.step(closure) # Optimizer update  
            else:
                loss = closure()
                self.global_optimizer.step() # Optimizer update  


        with torch.no_grad():
            for global_param, local_param in zip(self.global_net.parameters(), net.parameters()):
                local_param.data.copy_(global_param.data, non_blocking=True)
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

    def define_coarse_network(self, global_optimizer_type, args):
        # global network
        num_coarse_sbds = dist.get_world_size()

        if(global_optimizer_type==PrecLBFGS):
            coarse_optimizer = global_optimizer_type(self.global_net.extract_coarse_trainable_params(num_subdomains=num_coarse_sbds, overlap_width=0))
        else:
            coarse_optimizer = global_optimizer_type(self.global_net.extract_coarse_trainable_params(num_subdomains=num_coarse_sbds, overlap_width=0), lr=args.lr_global)
            # print("TODO:: add coarse nets  ")
            # exit(0)
        return coarse_optimizer

    def define_local_networks(self, net, local_optimizer_type, args):
        # local networks
        local_optimizer = local_optimizer_type(net.extract_trainable_params(sbd_id=dist.get_rank(), num_subdomains=dist.get_world_size(), overlap_width=args.overlap_width))
        self.scale = net.get_avg()
        dist.all_reduce(self.scale, op=dist.ReduceOp.SUM)
        return local_optimizer

    def train(self, dataset, net, criterion, global_optimizer_type, local_optimizer_type, args,  regularization=None, lr_scheduler_data=None, lr_scheduler_data_local=None):
        if self.use_cuda:
            net.cuda()
            torch.backends.cudnn.benchmark = True

        # broadcasting initial parameters to all nodes, so all nets have same configuration
        with torch.no_grad():
            for p in net.parameters():
                dist.broadcast(p, src=0)

        # global network
        self.global_net, self.global_optimizer = self.define_global_network(net, global_optimizer_type, args)

        # local network
        self.local_optimizers = self.define_local_networks(net, local_optimizer_type, args)
        
        # coarse network
        self.coarse_optimizer = self.define_coarse_network(global_optimizer_type, args)


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


            train_loss, pde_loss, train_acc = self.train_step(epoch, dataset, net, criterion, global_optimizer_type, local_optimizer_type, regularization, args)
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
            self.print_epoch(epoch, pde_loss, train_loss, loss_test, L2_error, L2_error_rel,  time_current, args)
            

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
    def compute_num_loss_grad_evals(self, num_loss_evals, epoch, args):

        if(args.use_SPQN):
            # additive
            num_loss_evals                 +=  self.global_optimizer.num_fun_evals
            self.global_optimizer.num_fun_evals = 0.0
            num_loss_evals                 +=  self.local_optimizers.num_fun_evals
            self.local_optimizers.num_fun_evals = 0.0

            # TODO:: change based on freq
            if(self.stochastic_update):
                num_grad_evals  = (2 * (epoch/args.freq)) + (epoch * args.num_local_steps)
            else:
                num_grad_evals  = (2 * epoch) + (epoch * args.num_local_steps)

            # scaling works as 1 CP per subdomain
            if self.use_coarse:
                num_loss_evals += self.coarse_optimizer.num_fun_evals* (self.num_subdomains/self.global_net.get_num_layers())
                self.coarse_optimizer.num_fun_evals = 0.0
                num_grad_evals += epoch * args.num_coarse_steps*(self.num_subdomains/self.global_net.get_num_layers())
        else:
            num_loss_evals +=  self.global_optimizer.num_fun_evals
            self.global_optimizer.num_fun_evals = 0.0
            num_grad_evals  = epoch


        return num_loss_evals, num_grad_evals

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

    def get_loss_grad(self, dataset, criterion, args): 
        self.global_optimizer.zero_grad()
        accum_loss = 0

        for batch_idx, (coordinates) in enumerate(dataset.train_loader):
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

            loss=loss/args.num_batches
            accum_loss += loss
            loss.backward()

        grad  = self.global_optimizer._gather_flat_grad() 

        return accum_loss, grad

    @torch.no_grad()
    def collect_local_corrections(self, net): 
        corr = [p_local - p for p, p_local in zip(self.global_net.parameters(), net.parameters())]
        correction = self.global_optimizer._gather_flat(corr)
        dist.all_reduce(correction, op=dist.ReduceOp.SUM)
        return correction / self.scale

    def local_to_global_correction(self, closure_global, net, criterion, dataset, args): 
        if(self.layerwise_lr_update):
            with torch.no_grad():
                corr = [p_local - p for p, p_local in zip(self.global_net.parameters(), net.parameters())]
                corr = self.global_optimizer._gather_flat(corr)
                correction = [torch.zeros_like(corr) for _ in range(dist.get_world_size())]
                dist.all_gather(correction, corr)

            # do sequential lr update on root processor
            loss_global, flat_grad  = self.get_loss_grad(dataset, criterion, args)
            for sbd_id in range(self.num_subdomains):
                loss_global, flat_grad = self.compute_lr_local_to_global(closure_global, correction[sbd_id], flat_grad, loss_global)
        else:
            correction = self.collect_local_corrections(net)
            loss_global_old, flat_grad_old  = self.get_loss_grad(dataset, criterion, args)
            loss_global_new, flat_grad_new = self.compute_lr_local_to_global(closure_global, correction, flat_grad_old, loss_global_old)



    def coarse_step(self, dataset, criterion, num_coarse_steps, args):
        for coarse_it in range(0, num_coarse_steps):

            # with accumulated gradients 
            def closure_coarse():
                accum_loss = 0
                self.coarse_optimizer.zero_grad()
                
                for batch_idx, (coordinates) in enumerate(dataset.train_loader):
                    if(self.training_pinn_flg == True):
                            if self.use_cuda:
                                coordinates = coordinates.cuda()  
                            coordinates.requires_grad=True                
                            __, loss_coarse, pde_loss_loc = criterion(self.global_net, coordinates)
                    else:
                        inputs, targets = coordinates
                        if self.use_cuda:
                            inputs = inputs.cuda()
                            targets = targets.cuda()

                        u_net = self.global_net(inputs)
                        loss_coarse = criterion(u_net, targets)

                    loss_coarse=loss_coarse/args.num_batches
                    accum_loss += loss_coarse
                    loss_coarse.backward()  # Backward Propagation
                
                return accum_loss

            loss_coarse = self.coarse_optimizer.step(closure_coarse) # Optimizer update



    def local_step(self, net, dataset, criterion, args):
        for local_it in range(0, args.num_local_steps):

            def closure_local():
                accum_loss = 0
                self.local_optimizers.zero_grad()
                
                for batch_idx, (coordinates) in enumerate(dataset.train_loader):
                    if(self.training_pinn_flg):
                        if self.use_cuda:
                            coordinates = coordinates.cuda()  
                        coordinates.requires_grad=True                
                        __, loss_loc, pde_loss_loc = criterion(net, coordinates)
                    else:
                        inputs, targets = coordinates
                        if self.use_cuda:
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                        
                        u_net = net(inputs)
                        loss_loc = criterion(u_net, targets)

                    loss_loc    =   loss_loc/args.num_batches
                    accum_loss  +=  loss_loc
                    loss_loc.backward()  # Backward Propagation

                return accum_loss

            loss_loc = self.local_optimizers.step(closure_local) # Optimizer update


    # TODO:: check if there are more efficient ways to load data from file !!!
    def train_step(self, epoch, dataset, net, criterion, global_optimizer_type, local_optimizer_type, regularization, args):
        loss_no_reg = 0
        loss_with_reg = 0
        loss=0
        total = 0
        state_global_step=0
        train_acc=0
        correct = 0


        if(self.global_opt_initialized==False or args.use_SPQN==False): 
            self.init_global_optim(net, dataset, criterion, global_optimizer_type, args)


        if(args.use_SPQN==True):
            start_time = time.time()


            if(self.use_coarse):
                # coarse minimization
                self.coarse_step(dataset, criterion, args.num_coarse_steps, args)

            # USUAL:: copy global to local
            with torch.no_grad():
                for param_global, param_local in zip(self.global_net.parameters(), net.parameters()):
                    param_local.data.copy_(param_global.data)


            # local minimization
            self.local_step(net, dataset, criterion, args)

            # # # # # # # # # # # # # # # get correction from local to global  # # # # # # # # # # # # # # # # # 
            def closure_global():
                accum_loss = 0
                self.global_optimizer.zero_grad()

                for batch_idx, (coordinates) in enumerate(dataset.train_loader):
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

                    loss=loss/args.num_batches
                    accum_loss += loss
                    loss.backward()  # Backward Propagation

                return accum_loss


            self.local_to_global_correction(closure_global, net, criterion, dataset, args)

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
        # if(epoch%1==0):
            pde_loss = torch.tensor(0.0)
            accum_loss = 0
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
                
                accum_loss += loss


            if(self.type == "classification"):
                train_acc = 100.*correct/total
                # print("xxxxx TRAIN:  train_acc ", train_acc)
            

            # for both, classification and regression cases
            loss = accum_loss/args.num_batches

  
        # print("loss, ", loss)
        # print("loss_no_reg, ", loss_no_reg)
        # print("loss_with_reg, ", loss_with_reg)

        return loss.item(), pde_loss.item(), train_acc


    # TODO:: make sure it takes into account gradient accumulation, if test set is also batching 
    # => current version not super correct 
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
        if(dist.get_rank()==0):
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")
            print("Epoch.     #g_ev        #l_ev      L_inner(train)        L_sum(train)          L_sum(test)           L2_error            L2_error_rel          time ")
            print("----------------------------------------------------------------------------------------------------------------------------------------------------")

    def print_epoch(self, epoch, pde_loss,  train_loss, test_loss, L2_error, L2_error_rel, time_current, args):
        if(epoch % self.freq_print == 0):
            self.save_model(self.global_net, args.model_name)

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



    def save_model(self, net, name):

        if(dist.get_rank()==0):
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
    
            # make sure synch 
            # with torch.no_grad():
            #     for p in net.parameters():
            #         dist.broadcast(p, src=0)

            return True
        else:
            print("DeepOnet not found ... ")
            return False
            

