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

from src.base_trainer import *
from src.pmw.dataloaders import GeneralizedDistributedDataLoader
from src.pmw.model_handler import ModelHandler
from src.utils import closure, dprint


class Trainer(BaseTrainer):

    @staticmethod
    def get_default_config():
        C = BaseTrainer.get_default_config()
        # data chunks amount
        C.data_chunks_amount = 1
        C.momentum = 0.9

        return C

    def __init__(self, config, model, train_dataset):
        super().__init__(config, model, train_dataset)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

        train_loader = GeneralizedDistributedDataLoader(model_handler=config.model_handler, 
                                                        dataset=self.train_dataset, 
                                                        batch_size=config.batch_size, 
                                                        shuffle=False, 
                                                        num_workers=config.num_workers, 
                                                        pin_memory=True)

        criterion = nn.CrossEntropyLoss()

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
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
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
