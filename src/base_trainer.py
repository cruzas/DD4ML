"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

from src.utils import CfgNode as CN
from src.utils import closure, dprint


class BaseTrainer(ABC):
    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 1
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0 
        C.epochs = 1 # in case epochs instead of iter
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
        self.model = self.model.to(self.device)
        dprint(f"running on device {self.device}")

        # variables that will be assigned to trainer class later for logging and etc
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
    
    def compute_epoch_loss(self):
        model.train()
        model, config = self.model, self.config
        criterion = self.criterion
        self.loss = 0.0 # epoch loss
        for batch_idx, (x, y) in enumerate(self.train_loader):    
            x, y = x.to(self.device), y.to(self.device)
            
            if self.epoch_num == 0:
                first_closure = closure(x, y, criterion, model, data_chunks_amount=config.data_chunks_amount, compute_grad=False)
                self.loss += first_closure()
            else:
                general_closure = closure(x, y, criterion=criterion, model=model, data_chunks_amount=config.data_chunks_amount, compute_grad=True, grad_norm_clip=config.grad_norm_clip)   
                
                # Check if final_subdomain_closure is part of self.optimizer arguments
                if hasattr(self.optimizer, 'final_subdomain_closure'):
                    def final_subdomain_closure(outputs, y=y):
                        y_chunks = y.chunk(len(outputs))
                        loss = []
                        for i, o in enumerate(outputs):
                            loss.append(self.criterion(o, y_chunks[i]))
                        return loss
                
                    self.loss += self.optimizer.step(closure=general_closure, final_subdomain_closure=final_subdomain_closure)
                else:
                    self.loss += self.optimizer.step(closure)
                
            # Print progress within the epoch
            self.epoch_progress = 100.0 * (batch_idx + 1) / total_batches
            self.trigger_callbacks('on_batch_end')
        
        self.loss = self.loss / total_batches
        tnow = time.time()
        self.epoch_dt = tnow - self.epoch_time
        self.epoch_time = tnow
    
    def compute_accuracy(self):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
        self.accuracy =  100.0 * correct / total
    
    def run_by_epoch(self):
        model, config = self.model, self.config
        train_loader = self.train_loader
        
        self.total_start_time = time.time()
        self.epoch_num = 0
        self.epoch_time = time.time()
        total_batches = len(train_loader)
        for epoch in range(config.epochs+1):
            self.compute_epoch_loss()
            self.compute_accuracy()
            self.running_time = time.time() - self.total_start_time
            self.trigger_callbacks('on_epoch_end')
            self.epoch_num += 1
    
    def run_by_iter(self):
        model.train()
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        
        for self.iter_num in range(config.max_iters if config.max_iters is not None else float('inf')):
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
        
            if self.iter_num == 0:
                first_closure = closure(x, y, criterion, model, data_chunks_amount=config.data_chunks_amount, compute_grad=False)
                self.loss = first_closure()
            else:
                self.optimizer.zero_grad()      
                general_closure = closure(x, y, criterion=criterion, model=model, data_chunks_amount=config.data_chunks_amount, compute_grad=True, grad_norm_clip=config.grad_norm_clip)        
                self.loss = self.optimizer.step(closure=general_closure)
            
            self.trigger_callbacks('on_batch_end')
            
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

