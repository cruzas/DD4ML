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
from src.pmw.dataloaders import GeneralizedDistributedDataLoader
from src.pmw.model_handler import ModelHandler
from src.utils import closure, dprint


class Trainer(BaseTrainer):

    @staticmethod
    def get_default_config():
        C = BaseTrainer.get_default_config()
        C.data_chunks_amount = 1
        C.use_pmw = False
        C.run_by_epoch = True # if False, run by iteration

        return C

    def __init__(self, config, model, optimizer, criterion, train_dataset, test_dataset):
        super().__init__(config, model, optimizer, criterion, train_dataset, test_dataset)

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
