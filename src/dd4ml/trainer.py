"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

Note:
This code runs with our dictionary-defined model, which is instantiated as a ParallelizedModel object.
Model handler takes care of the parallelized model logic. This is why this is slightly different from the trainer in mingpt.
"""

import inspect
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dd4ml.pmw.dataloaders import GeneralizedDistributedDataLoader
from dd4ml.utility import CfgNode as CN
from dd4ml.utility import closure


class Trainer:

    @staticmethod
    def get_default_config():
        # Base settings
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        # optimizer parameters
        C.max_iters = 1000
        C.batch_size = 128
        C.learning_rate = 5e-4
        C.betas = (0.9, 0.999)  # for Adam
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.epochs = 3  # in case epochs instead of iter
        C.run_by_epoch = (
            False  # if False, run by iteration, typically for transformer networks
        )

        # For APTS_D
        C.correct_step = False  # for APTS_D
        C.norm_type = 2  # for APTS_D (and possibly APTS)
        C.ema = True  # for APTS_D
        C.global_pass=False
        C.foc=False 
        
        # For APTS* 
        C.max_global_iters = 1  # for APTS*
        C.max_subdomain_iters = 3  # for APTS*
        C.global_second_order = False  # for APTS*
        C.local_second_order = False  # for APTS*
        C.subdomain_optimizer = None
        C.gradient_accumulation = True
        C.accumulation_steps = 5
        
        # For pipelining via pwm library
        C.data_chunks_amount = 1
        C.use_pmw = False

        return C

    def __init__(
        self, config, model, optimizer, criterion, train_dataset, test_dataset
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = (
                f"cuda:{torch.cuda.current_device()}"
                if dist.get_backend() != "gloo"
                else "cpu"
            )
        else:
            self.device = config.device

        # In case we are on a Mac with MPS enabled, we can use it as a device
        if (
            self.device == "cpu"
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
            and (
                (dist.is_initialized() and dist.get_world_size() == 1)
                or not dist.is_initialized()
            )
        ):
            self.device = torch.device("mps")

        self.model = self.model.to(self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.total_start_time = 0.0  # for computing the total running time
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
            train_loader = GeneralizedDistributedDataLoader(
                model_handler=config.model_handler,
                dataset=self.train_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        else:
            # Initialize world_size
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            # Global batch size from config
            global_batch_size = config.batch_size  # e.g., 200

            # Per-process batch size
            per_process_batch_size = global_batch_size // world_size

            train_sampler = DistributedSampler(self.train_dataset)
            test_sampler = DistributedSampler(self.test_dataset)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=per_process_batch_size,  # Use per-process batch size
                sampler=train_sampler,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            test_loader = DataLoader(
                self.test_dataset,
                batch_size=per_process_batch_size,
                sampler=test_sampler,
                num_workers=config.num_workers,
                pin_memory=True,
            )

        self.train_loader = train_loader
        self.test_loader = test_loader

        if config.run_by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()

    def compute_epoch_loss(self):
        model, config = self.model, self.config

        model.train()
        criterion = self.criterion
        self.loss = 0.0  # epoch loss
        total_batches = len(self.train_loader)
        self.iter_num = 0
        for batch_idx, (x, y) in enumerate(self.train_loader):
            batch_time_start = time.time()
            x, y = x.to(self.device), y.to(self.device)

            if self.epoch_num == 0:
                first_closure = closure(
                    x,
                    y,
                    criterion,
                    model,
                    data_chunks_amount=config.data_chunks_amount,
                    compute_grad=False,
                )
                self.loss += first_closure()
            else:
                general_closure = closure(
                    x,
                    y,
                    criterion=criterion,
                    model=model,
                    data_chunks_amount=config.data_chunks_amount,
                    compute_grad=True,
                    grad_norm_clip=config.grad_norm_clip,
                )

                # Check if final_subdomain_closure is part of self.optimizer arguments
                if (
                    "final_subdomain_closure"
                    in inspect.signature(self.optimizer.step).parameters
                ):

                    def final_subdomain_closure(outputs, y=y):
                        y_chunks = y.chunk(len(outputs))
                        loss = []
                        for i, o in enumerate(outputs):
                            loss.append(self.criterion(o, y_chunks[i]))
                        return loss

                    self.loss += self.optimizer.step(
                        closure=general_closure,
                        final_subdomain_closure=final_subdomain_closure,
                    )
                elif "apts_d" in self.optimizer.__class__.__name__.lower():
                    self.loss += self.optimizer.step(inputs=x, labels=y)
                else:
                    self.loss += self.optimizer.step(closure=general_closure)

            # Print progress within the epoch
            self.epoch_progress = 100.0 * (batch_idx + 1) / total_batches
            self.batch_dt = time.time() - batch_time_start
            self.iter_dt = self.batch_dt
            self.running_time = time.time() - self.total_start_time
            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1

        self.loss = self.loss / total_batches
        tnow = time.time()
        self.epoch_dt = tnow - self.epoch_time
        self.epoch_time = tnow

    def compute_accuracy(self):
        model = self.model
        model.eval()
        correct = 0
        total = 0

        cond_std = not hasattr(model, "model_handler")
        cond_d_a = (
            hasattr(model, "model_handler") and model.model_handler.is_last_stage()
        )
        cond_d_b = False
        if not cond_std:
            first_last_stage_rank = model.model_handler.get_stage_ranks(
                stage_name="last", mode="global"
            )[0]
            cond_d_b = dist.get_rank() == first_last_stage_rank
        with torch.no_grad():
            for _, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)

                if not cond_std:
                    assert len(outputs) == 1
                    outputs = outputs[0]

                if cond_std or (cond_d_a and cond_d_b):
                    outputs = outputs.to(x.device)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            if not cond_std:
                # Broadcast correct and total to all ranks
                correct = torch.tensor(correct, device=model.tensor_device)
                total = torch.tensor(total, device=model.tensor_device)
                dist.broadcast(correct, src=first_last_stage_rank, async_op=False)
                dist.broadcast(total, src=first_last_stage_rank, async_op=False)
                correct = correct.item()
                total = total.item()

            self.accuracy = 100.0 * correct / total

    def run_by_epoch(self):
        config = self.config

        self.total_start_time = time.time()
        self.epoch_num = 0
        self.epoch_time = time.time()
        for _ in range(config.epochs + 1):
            self.compute_epoch_loss()
            self.compute_accuracy()
            self.running_time = time.time() - self.total_start_time
            self.trigger_callbacks("on_epoch_end")
            self.epoch_num += 1

    def run_by_iter(self):
        model, config = self.model, self.config

        model.train()
        self.total_start_time = time.time()
        self.iter_time = time.time()
        self.data_iter = iter(self.train_loader)
        for self.iter_num in range(
            config.max_iters + 1 if config.max_iters is not None else float("inf")
        ):
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                batch = next(self.data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            if self.iter_num == 0:
                first_closure = closure(
                    x,
                    y,
                    self.criterion,
                    self.model,
                    data_chunks_amount=config.data_chunks_amount,
                    compute_grad=False,
                )
                self.loss = first_closure()
            else:
                self.optimizer.zero_grad()
                general_closure = closure(
                    x,
                    y,
                    criterion=self.criterion,
                    model=self.model,
                    data_chunks_amount=config.data_chunks_amount,
                    compute_grad=True,
                    grad_norm_clip=config.grad_norm_clip,
                )
                # Check if self.optimizer.step requires final_subdomain_closure
                if (
                    "final_subdomain_closure"
                    in inspect.signature(self.optimizer.step).parameters
                ):  # for APTS

                    def final_subdomain_closure(outputs, y=y):
                        y_chunks = y.chunk(len(outputs))
                        loss = []
                        for i, o in enumerate(outputs):
                            loss.append(self.criterion(o, y_chunks[i]))
                        return loss

                    self.loss = self.optimizer.step(
                        closure=general_closure,
                        final_subdomain_closure=final_subdomain_closure,
                    )
                elif "apts_d" in self.optimizer.__class__.__name__.lower():
                    self.loss += self.optimizer.step(inputs=x, labels=y)
                else:
                    self.loss = self.optimizer.step(closure=general_closure)

            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.running_time = tnow - self.total_start_time
            self.trigger_callbacks("on_batch_end")
            self.iter_time = tnow

    def compute_perplexity(self):
        pass
