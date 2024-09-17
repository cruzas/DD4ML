import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
import sys
import numpy as np
import time
import os
import sys
import pandas as pd
from trainers.Config import *
import torch.optim as optim
from trainers.PrecLBFGS import *
from trainers.GDLinesearch import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from optimizers import TR

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


class UnconstrainedRegressionTrainer(object):
    # def __init__(self):
    #     self.config = copy.deepcopy(trainer_config)
    #     self.use_cuda = torch.cuda.is_available()

    def __init__(self, training_pinn_flg, num_subdomains=1, type="regression", freq_print=1, use_wandb=False):
        self.config = copy.deepcopy(trainer_config)
        self.use_cuda = torch.cuda.is_available()
        self.training_pinn_flg = training_pinn_flg
        self.num_subdomains = 1
        self.type = type

        self.best_acc_train = 0.0
        self.best_acc_test = 0.0

        self.freq_print = freq_print
        self.use_wandb = use_wandb

        self.convergence_status = dict( epochs=0,
                                        num_loss_evals=0,
                                        loss_train=0, 
                                        loss_val=0, 
                                        L2_error=0, 
                                        L2_error_rel=0)


    def train(self, dataset, net, criterion, optimizer_type, args, regularization=None, lr_scheduler=None):

        # if self.use_cuda:
        #     net.cuda()
        #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        #     torch.backends.cudnn.benchmark = True

        if(optimizer_type == optim.SGD):
            optimizer = optimizer_type(net.parameters(), lr=args.lr_global, momentum=0.9)
        elif(optimizer_type == optim.Adam):
            optimizer = optimizer_type(net.parameters(), lr=args.lr_global)            
        elif(optimizer_type == PrecLBFGS):
            optimizer = optimizer_type(net.parameters())
        else:
            optimizer = optimizer_type(model= net, lr=args.lr_global, subdomain_optimizer=optim.SGD, subdomain_optimizer_defaults={'lr': args.lr_global}, global_optimizer=TR, global_optimizer_defaults={'lr': args.lr_global}, max_subdomain_iter=5, dogleg=True, APTS_in_data_sync_strategy='average', step_strategy='mean')

        epoch = 0
        self.best_test_loss = 9e9
        self.best_loss = 9e9

        self.L2_error       = 9e9
        self.L2_error_rel   = 9e9        

        # if(self.config["resume_from_checkpoint"]==False):
        #    net.init_params()
        # else:
        #    checkpoint = torch.load(self.config["checkpoint_folder_name"] + os.sep  + self.config["checkpoint_file_name"]+'.t7')
        #    net = checkpoint['net']
        #    epoch = checkpoint['epoch'] + 1

        self.print_init()
        elapsed_time = 0
        converged = False

        if(lr_scheduler is not None):
            sched_type, milestones_user, gamma_user = lr_scheduler
            lr_scheduler = sched_type(optimizer, milestones_user, gamma_user)
        else:
            lr_scheduler = None

        while(converged == False):
            start_time = time.time()
            pde_loss, train_loss = self.train_step(
                dataset, net, criterion, optimizer, regularization)
            test_loss, self.L2_error, self.L2_error_rel = self.test_step(
                dataset, net, criterion, optimizer, epoch)


            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            epoch += 1

            time_current = str('%d:%02d:%02d' % (get_hms(elapsed_time)))

            self.print_epoch(epoch, pde_loss, train_loss,
                             test_loss, self.L2_error, self.L2_error_rel, time_current)

            if(lr_scheduler is not None):
                lr_scheduler.step()

            if(epoch >= self.config["max_epochs"]):
                converged = True
                print("converged max_epochs")
            elif(np.isfinite(pde_loss) == False or np.isfinite(train_loss) == False or np.isfinite(test_loss) == False):
                converged = True
            elif(train_loss < self.config["loss_train_tol"] or test_loss < self.config["loss_test_tol"]):
                print("converged small loss")
                converged = True


        num_loss_evals = 0
        if(optimizer_type == PrecLBFGS or optimizer_type == GDLinesearch):
            num_loss_evals = optimizer.num_fun_evals


        self.convergence_status["epochs"]           = epoch
        self.convergence_status["num_loss_evals"]   = num_loss_evals
        self.convergence_status["num_grad_evals"]   = epoch


        self.convergence_status["loss_train"]   = self.best_loss
        self.convergence_status["loss_val"]     = self.best_test_loss
        
        self.convergence_status["L2_error"]     = self.L2_error
        self.convergence_status["L2_error_rel"] = self.L2_error_rel



        if(self.config.get("verbose")==True):
            print("self.convergence_status  ", self.convergence_status)



        return self.convergence_status


    def train_step(self, dataset, net, criterion, optimizer, regularization):
        net.train()
        net.training = True
        train_loss = 0
        pde_loss_sum = 0
        loss = 0
        pde_loss = 0

        for batch_idx, (coordinates) in enumerate(dataset.train_loader):

            if(self.training_pinn_flg == False):
                inputs, targets = coordinates
                if self.use_cuda:
                    inputs = inputs.cuda()                
                    targets = targets.cuda()
            else:
                inputs = coordinates[0]
                if self.use_cuda:
                    inputs = inputs.cuda()     
                try:
                    inputs.requires_grad = True
                except:
                    print('asd')


            def closure(**kwargs):
                nonlocal loss
                nonlocal pde_loss
                nonlocal inputs
                nonlocal targets

                optimizer.zero_grad()

                if(self.training_pinn_flg == False):
                    u_net = net(inputs)
                    loss = criterion(u_net, targets)
                    pde_loss = loss
                else:
                    __, loss, pde_loss = criterion(net, inputs)

                if(regularization is not None):
                    reg_loss = regularization(net)
                    loss += reg_loss

                loss.backward()  # Backward Propagation
                return loss

            optimizer.step(closure)  # Optimizer update

            train_loss += loss.item()
            pde_loss_sum += pde_loss.item()

        return pde_loss_sum/len(dataset.train_loader), train_loss/len(dataset.train_loader)



    def test_step(self, dataset, net, criterion, optimizer, epoch):

        net.eval()
        net.training = False
        test_loss = 0
        test_pde_loss = 0
        test_acc = 0
        l2_error = 0
        l2_error_rel = 0

        # with torch.no_grad():
        for batch_idx, (coordinates) in enumerate(dataset.test_loader):
            if(self.training_pinn_flg == False):
                inputs, targets = coordinates
                
                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets   = targets.cuda()

            else:
                coordinates_sol = coordinates[0]

                coordinates = coordinates_sol[:, 0:-1]
                exact_sol   = coordinates_sol[:, -1]


                if self.use_cuda:
                    coordinates = coordinates.cuda()
                    exact_sol   = exact_sol.cuda()

                coordinates.requires_grad = True



            if(self.training_pinn_flg == False):
                u_net = net(coordinates)
                loss = criterion(u_net, targets)
            else:
                u_net, loss, loss_inner = criterion(net, coordinates)

                
                l2_error        += torch.norm(exact_sol - u_net).item()
                l2_error_rel    += (l2_error/torch.norm(exact_sol)).item()


            test_loss += loss.item()

            if(self.type == "classification"):
                _, predicted = torch.max(u_net.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).sum()

        test_loss /= len(dataset.test_loader)
        # l2_error  /= len(dataset.test_loader)

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_loss = test_loss

        if(self.type == "classification"):
            test_acc = 100.*correct/total

        if test_acc < self.best_acc_test:
            self.best_acc_test = test_acc
            self.best_acc = acc_train
         

        if l2_error_rel < self.L2_error_rel:
            self.L2_error     = l2_error
            self.L2_error_rel = l2_error_rel   


        #  optimizer.zero_grad()

        if(self.config["checkpoint_folder_name"] is not None):
            if test_loss < self.best_test_loss:
                print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (test_loss))
                state = {
                    'net': net.module if self.use_cuda else net,
                    'loss': test_loss,
                    'L2_error': self.L2_error,
                    'L2_error_rel': self.L2_error_rel,
                    'epoch': epoch,
                }

                if not os.path.isdir(self.config["checkpoint_folder_name"]):
                    os.mkdir(self.config["checkpoint_folder_name"])
                torch.save(state, self.config["checkpoint_folder_name"] +
                           os.sep + self.config["checkpoint_file_name"]+'.t7')
                self.best_test_loss = test_loss

        return test_loss, l2_error, l2_error_rel


    def print_init(self):
        print("----------------------------------------------------------------------------------------------------------------------------")
        print("Epoch      L_inner(train)        L_sum(train)          L_sum(test)           L2_error            L2_error_rel          time ")
        print("----------------------------------------------------------------------------------------------------------------------------")



    def print_epoch(self, epoch, pde_loss,  train_loss, test_loss, L2_error, L2_error_rel, time_current):
        if(epoch % self.freq_print == 0):
            print("{:03d}         {:e}         {:e}         {:e}         {:e}         {:e}        {:s} ".format(
                epoch, pde_loss, train_loss, test_loss, L2_error, L2_error_rel, time_current))

            if(self.config.get("conv_history_csv_name") is not None):
                df = pd.DataFrame({'epoch': [epoch],
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



if __name__ == '__main__':
    trainer = UnconstrainedRegressionTrainer()





