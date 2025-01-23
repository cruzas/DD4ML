"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

Note:
This code runs with our dictionary-defined model, which is instantiated as a ParallelizedModel object.
Model handler takes care of the parallelized model logic. This is why this is slightly different from the trainer in mingpt.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.base_trainer import *
from src.pmw.model_handler import ModelHandler
from src.utils import dprint


class Trainer(BaseTrainer):

    @staticmethod
    def get_default_config():
        C = BaseTrainer.get_default_config()
        # data chunks amount
        C.data_chunks_amount = 1
        C.momentum = 0.9

        return C

    def __init__(self, config, model, train_dataset, test_dataset=None):
        super().__init__(config, model, train_dataset, test_dataset=test_dataset)
        self.model.to(self.device)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=config.world_size, rank=config.rank)
        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

        criterion = nn.CrossEntropyLoss().to(self.device)

        self.epoch_num = 0
        self.epoch_time = time.time()
        total_batches = len(train_loader)
        for epoch in range(config.num_epochs+1):
            model.train()
            self.loss = 0.0 # running loss
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                def closure():
                    self.optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    return loss
                
                if self.epoch_num == 0:
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, targets)
                    self.loss += batch_loss
                else:   
                    self.loss += self.optimizer.step(closure)

                # Print progress within the epoch
                self.epoch_progress = 100.0 * (batch_idx + 1) / total_batches
                self.trigger_callbacks('on_batch_end')
            
            self.loss = self.loss / total_batches
            tnow = time.time()
            self.epoch_dt = tnow - self.epoch_time
            self.epoch_time = tnow

            # compute accuracy
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                
            self.accuracy =  100.0 * correct / total
            self.trigger_callbacks('on_epoch_end')
            self.epoch_num += 1
