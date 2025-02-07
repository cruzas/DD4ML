"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

Note:
This code runs with our dictionary-defined model, which is instantiated as a ParallelizedModel object.
Model handler takes care of the parallelized model logic. This is why this is slightly different from the trainer in mingpt.
"""
import inspect
import time
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader

from src.pmw.dataloaders import GeneralizedDistributedDataLoader
from src.pmw.model_handler import ModelHandler
from src.utils import CfgNode as CN
from src.utils import closure, dprint


class Trainer():

    @staticmethod
    def get_default_config():
        # Base settings
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 1
        # optimizer parameters
        C.max_iters = 1000
        C.batch_size = 128
        C.learning_rate = 5e-4
        C.betas = (0.9, 0.999) # for Adam
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0 
        C.epochs = 3 # in case epochs instead of iter
        C.run_by_epoch = False # if False, run by iteration, typically for transformer networks
        
        # For pipelining via pwm library
        C.data_chunks_amount = 1
        C.use_pmw = False

        return C

    def __init__(self, config, model, optimizer, criterion, train_dataset, test_dataset):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'   
        else:
            self.device = config.device
        
        # In case we are on a Mac with MPS enabled, we can use it as a device
        if self.device == 'cpu' and torch.backends.mps.is_available() and torch.backends.mps.is_built() and ((dist.is_initialized() and dist.get_world_size() == 1) or not dist.is_initialized()):
            self.device = torch.device("mps")     
             
        self.model = self.model.to(self.device)
        dprint(f"running on device {self.device}")

        # variables that will be assigned to trainer class later for logging and etc
        self.total_start_time = 0.0 # for computing the total running time
        if config.run_by_epoch:
            self.epoch_num = 0
            self.epoch_time = 0.0
            self.epoch_dt = 0.0
        else:
            self.iter_num = 0
            self.iter_time = 0.0
            self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        _, config = self.model, self.config
        
        if config.use_pmw:
            self.train_loader = GeneralizedDistributedDataLoader(model_handler=config.model_handler, 
                                                            dataset=self.train_dataset, 
                                                            batch_size=config.batch_size, 
                                                            shuffle=False, 
                                                            num_workers=config.num_workers, 
                                                            pin_memory=True)
            self.test_loader = GeneralizedDistributedDataLoader(model_handler=config.model_handler,
                                                           dataset=self.test_dataset, 
                                                           batch_size=config.batch_size, 
                                                           shuffle=False, 
                                                           num_workers=config.num_workers, 
                                                           pin_memory=True)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
            self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

        if config.run_by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()

    def compute_epoch_loss(self):
        model, config = self.model, self.config
        
        model.train()
        criterion = self.criterion
        self.loss = 0.0 # epoch loss
        total_batches = len(self.train_loader)
        for batch_idx, (x, y) in enumerate(self.train_loader):
            batch_time_start = time.time()   
            x, y = x.to(self.device), y.to(self.device)
            
            if self.epoch_num == 0:
                first_closure = closure(x, y, criterion, model, data_chunks_amount=config.data_chunks_amount, compute_grad=False)
                self.loss += first_closure()
            else:
                general_closure = closure(x, y, criterion=criterion, model=model, data_chunks_amount=config.data_chunks_amount, compute_grad=True, grad_norm_clip=config.grad_norm_clip)   
                
                # Check if final_subdomain_closure is part of self.optimizer arguments
                if 'final_subdomain_closure' in inspect.signature(self.optimizer.step).parameters:
                    def final_subdomain_closure(outputs, y=y):
                        y_chunks = y.chunk(len(outputs))
                        loss = []
                        for i, o in enumerate(outputs):
                            loss.append(self.criterion(o, y_chunks[i]))
                        return loss
                
                    self.loss += self.optimizer.step(closure=general_closure, final_subdomain_closure=final_subdomain_closure)
                else:
                    self.loss += self.optimizer.step(closure=general_closure)
                
            # Print progress within the epoch
            self.epoch_progress = 100.0 * (batch_idx + 1) / total_batches
            self.batch_dt = time.time() - batch_time_start
            self.trigger_callbacks('on_batch_end')
        
        self.loss = self.loss / total_batches
        tnow = time.time()
        self.epoch_dt = tnow - self.epoch_time
        self.epoch_time = tnow
    
    def compute_accuracy(self):
        model, config = self.model, self.config

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                # Check if outputs is a list of length 1
                if len(outputs) == 1:
                    outputs = outputs[0]
                outputs = outputs.to(x.device)

                # Handle the case where outputs is a list (e.g., due to chunked data)
                if isinstance(outputs, list):
                    outputs = torch.stack(outputs).mean(dim=0)  # Average over chunks

                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        self.accuracy = 100.0 * correct / total
    
    def run_by_epoch(self):
        model, config = self.model, self.config
        
        self.total_start_time = time.time()
        self.epoch_num = 0
        self.epoch_time = time.time()
        for epoch in range(config.epochs+1):
            self.compute_epoch_loss()
            self.compute_accuracy()
            self.running_time = time.time() - self.total_start_time
            self.trigger_callbacks('on_epoch_end')
            self.epoch_num += 1
    
    def run_by_iter(self):
        model, config = self.model, self.config
        
        model.train()
        self.total_start_time = time.time()
        self.iter_time = time.time()
        self.data_iter = iter(self.train_loader)
        for self.iter_num in range(config.max_iters+1 if config.max_iters is not None else float('inf')):
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                batch = next(self.data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
        
            if self.iter_num == 0:
                first_closure = closure(x, y, self.criterion, self.model, data_chunks_amount=config.data_chunks_amount, compute_grad=False)
                self.loss = first_closure()
            else:
                self.optimizer.zero_grad()      
                general_closure = closure(x, y, criterion=self.criterion, model=self.model, data_chunks_amount=config.data_chunks_amount, compute_grad=True, grad_norm_clip=config.grad_norm_clip)        
                # Check if self.optimizer.step requires final_subdomain_closure
                if 'final_subdomain_closure' in inspect.signature(self.optimizer.step).parameters: # for APTS
                    def final_subdomain_closure(outputs, y=y):
                        y_chunks = y.chunk(len(outputs))
                        loss = []
                        for i, o in enumerate(outputs):
                            loss.append(self.criterion(o, y_chunks[i]))
                        return loss
                    
                    self.loss = self.optimizer.step(closure=general_closure, final_subdomain_closure=final_subdomain_closure)
                else:
                    self.loss = self.optimizer.step(closure=general_closure)
            
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.running_time = tnow - self.total_start_time
            self.trigger_callbacks('on_batch_end')
            self.iter_time = tnow

    def compute_perplexity(self):
        pass   