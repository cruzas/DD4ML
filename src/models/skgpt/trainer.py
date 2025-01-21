"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from src.optimizers.apts import APTS
from src.optimizers.trust_region import TrustRegion
from src.pmw.dataloaders import GeneralizedDistributedDataLoader
from src.pmw.model_handler import ModelHandler
from src.utils import CfgNode as CN
from src.utils import closure, dprint


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        # the model we're using is so small that we can go a bit faster
        C.learning_rate = 1e-3
        C.betas = (0.9, 0.95) # in case of Adam optimizer used somewhere
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0 # max norm for gradient clipping
        # subdomain optimizer
        C.subdomain_optimizer = torch.optim.AdamW
        C.subdomain_optimizer_args = {'lr' : C.learning_rate}
        if C.subdomain_optimizer == torch.optim.AdamW:
            C.subdomain_optimizer_args['betas'] = C.betas
        elif C.subdomain_optimizer == torch.optim.SGD:
            C.subdomain_optimizer_args['momentum'] = 0.9
        C.max_subdomain_iters = 3
        # global optimizer
        C.global_optimizer = TrustRegion
        C.global_optimizer_args = {
            'max_lr': 1.0,
            'min_lr': 1e-5,
            'eta1': 0.25,
            'eta2': 0.75,
            'alpha1': 0.5,
            'alpha2': 2.0,
            'history_size': 5,
        }
        # data chunks amount
        C.data_chunks_amount = 1

        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
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

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        model.configure_params(config)
        self.optimizer = APTS(model=self.model,
                              subdomain_optimizer=config.subdomain_optimizer,
                              subdomain_optimizer_defaults=config.subdomain_optimizer_args,
                              global_optimizer=config.global_optimizer,
                              global_optimizer_defaults=config.global_optimizer_args,
                              lr=config.learning_rate,
                              max_subdomain_iter=config.max_subdomain_iters,
                              dogleg=True,
                              APTS_in_data_sync_strategy='average', 
                              step_strategy='mean'
                              )

        train_loader = GeneralizedDistributedDataLoader(model_handler=config.model_handler, 
                                                        dataset=self.train_dataset, 
                                                        batch_size=config.batch_size, 
                                                        shuffle=False, 
                                                        num_workers=config.num_workers, 
                                                        pin_memory=True)

        def criterion(logits, targets):
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

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
                def final_subdomain_closure(outputs, y=y):
                    y_chunks = y.chunk(len(outputs))
                    loss = []
                    for i, o in enumerate(outputs):
                        loss.append(criterion(o, y_chunks[i]))
                    return loss

                self.optimizer.zero_grad()      
                general_closure = closure(x, y, criterion=criterion, model=model, data_chunks_amount=config.data_chunks_amount, compute_grad=True)        
                self.loss = self.optimizer.step(closure=general_closure, final_subdomain_closure=final_subdomain_closure)
                
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
