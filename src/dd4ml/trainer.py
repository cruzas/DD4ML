"""
This code also runs with our dictionary-defined model, which is instantiated as a ParallelizedModel object.
Model handler takes care of the parallelized model logic. This is why this is slightly different from the trainer in mingpt.
"""

import inspect
import math
import os
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
from dd4ml.utility import CfgNode as CN
from dd4ml.utility import closure, dprint

from .dataloaders import GeneralizedDistributedDataLoader, OverlapBatchSampler


class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        # tolerance for convergence
        C.tol = 1e-6
        # device
        C.device = "auto"
        # data loader workers
        C.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        # training schedule
        C.epochs = 3
        C.run_by_epoch = False  # if False, run by iteration instead of epochs, typically for transformer networks
        C.max_iters = 1000
        # optimizer
        C.learning_rate = 1e-3
        C.betas = (0.9, 0.999)  # for Adam
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        # initial batch size and adaptive params
        C.batch_size = 128  # max batch size is lenght of dataset
        C.batch_inc_factor = 1  # factor to increase batch size
        C.loss_tol = 1e-3  # loss tolerance for adaptive batch size
        # APTS and TR
        C.delta = 0.1  # for trust region methods
        C.min_delta = 1e-3
        C.max_delta = 2.0
        C.data_parallel = False
        C.norm_type = 2  # for APTS_D (and possibly APTS_IP)
        C.glob_pass = False
        C.foc = False  # for APTS_D
        C.dogleg = False  # for APTS_D
        C.max_glob_iters = 1  # for APTS*
        C.max_loc_iters = 3  # for APTS*
        C.glob_second_order = False  # for APTS*
        C.loc_second_order = False  # for APTS*
        C.max_wolfe_iters = 10  # for APTS*
        C.gradient_accumulation = True  # for APTS*
        C.accumulation_steps = 1  # for APTS*
        C.mem_length = 3  # for TR methods
        # For pipelining via pwm library
        C.data_chunks_amount = 1
        C.use_pmw = False
        C.loc_opt = None
        C.glob_opt = None
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
            print("Doing auto device selection...")
            self.device = (
                f"cuda:{torch.cuda.current_device()}"
                if dist.get_backend() != "gloo"
                else "cpu"
            )
        else:
            self.device = config.device

        print(f"Is self.device == 'cpu'? {self.device == 'cpu'}")
        print(f"is torch.backends.mps available? {torch.backends.mps.is_available()}")
        print(f"is torch.backends.mps built? {torch.backends.mps.is_built()}")
        print(f"Is distributed initialized? {dist.is_initialized()}")
        print(f"World size: {dist.get_world_size() if dist.is_initialized() else 1}")

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
        # adaptive-batch state
        self.current_batch_size = config.batch_size
        self.max_batch_size_reached = False
        self.last_loss = float("inf")

        # timing
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

    def setup_data_loaders(self):
        """(Re)create train and test loaders using current_batch_size"""
        cfg, ds_train, ds_test = self.config, self.train_dataset, self.test_dataset
        bs = self.current_batch_size
        overlap = (
            cfg.overlap if hasattr(cfg, "overlap") else 0
        )  # Overlap between consecutive batches
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if cfg.use_pmw:
            self.train_loader = GeneralizedDistributedDataLoader(
                model_handler=cfg.model_handler,
                dataset=ds_train,
                batch_size=bs,
                shuffle=False,
                overlap=overlap,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

            self.test_loader = DataLoader(
                ds_test,
                batch_size=bs,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        else:
            # Per-process batch size
            pp_bs = bs // world_size

            base_train_sampler = DistributedSampler(
                ds_train,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            base_test_sampler = DistributedSampler(
                ds_test,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )

            train_sampler = OverlapBatchSampler(
                base_sampler=base_train_sampler,
                batch_size=pp_bs,
                overlap=overlap,
                drop_last=False,
            )

            test_sampler = OverlapBatchSampler(
                base_sampler=base_test_sampler,
                batch_size=pp_bs,
                overlap=overlap,
                drop_last=False,
            )

            self.train_loader = DataLoader(
                ds_train,
                batch_sampler=train_sampler,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

            self.test_loader = DataLoader(
                ds_test,
                batch_sampler=test_sampler,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

    def adjust_batch_size(self, loss):
        """Increase batch size when current loss is greater than previous loss."""
        cfg = self.config
        if cfg.batch_inc_factor == 1:
            return  # no batch size adjustment

        if loss > self.last_loss - cfg.loss_tol and not self.max_batch_size_reached:
            new_bs = min(
                int(math.floor(self.current_batch_size * cfg.batch_inc_factor)),
                len(self.train_dataset),
            )
            if new_bs == len(self.train_dataset):
                self.max_batch_size_reached = True
                dprint(
                    f"Current loss ({loss:.4f}) is greater than previous loss - tolerance ({(self.last_loss-cfg.loss_tol):.4f}). Batch size is already at maximum ({new_bs})."
                )
                return
            dprint(
                f"Current loss ({loss:.4f}) is greater than previous loss - tolerance ({(self.last_loss-cfg.loss_tol):.4f}). Increasing batch size from {self.current_batch_size} to {new_bs}."
            )
            self.current_batch_size = new_bs
            # Rebuild data loaders to reflect the new batch size
            self.setup_data_loaders()
            # Invalidate cached shapes for any WeightParallelizedTensor parameters
            for p in self.model.parameters():
                if isinstance(p, WeightParallelizedTensor):
                    p.invalidate_shape_cache()
        self.last_loss = loss

    def sample_control_batch(self, batch_size: int = 1):
        """Randomly sample a small control batch from the training dataset."""
        idx = torch.randint(0, len(self.train_dataset), (batch_size,))
        xs, ys = zip(*(self.train_dataset[int(i)] for i in idx))
        xs = torch.stack([torch.as_tensor(x) for x in xs])
        ys = torch.as_tensor(ys)
        return xs.to(self.device), ys.to(self.device)

    def run(self):
        self.setup_data_loaders()
        if self.config.run_by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()

    def compute_epoch_loss(self):
        model, cfg = self.model, self.config
        model.train()
        criterion = self.criterion
        total_loss = 0.0  # epoch loss
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
                    data_chunks_amount=cfg.data_chunks_amount,
                    compute_grad=False,
                )
                total_loss += first_closure()
            else:
                general_closure = closure(
                    x,
                    y,
                    criterion=criterion,
                    model=model,
                    data_chunks_amount=cfg.data_chunks_amount,
                    compute_grad=True,
                    grad_norm_clip=cfg.grad_norm_clip,
                )
                step_args = {}
                sig = inspect.signature(self.optimizer.step).parameters
                # Check if final_subdomain_closure is part of self.optimizer arguments
                if "final_subdomain_closure" in sig:

                    def final_subdomain_closure(outputs, y=y):
                        y_chunks = y.chunk(len(outputs))
                        return [
                            self.criterion(o, yc) for o, yc in zip(outputs, y_chunks)
                        ]

                    step_args = {
                        "closure": general_closure,
                        "final_subdomain_closure": final_subdomain_closure,
                    }
                elif any(
                    k in self.optimizer.__class__.__name__.lower()
                    for k in ("apts_d", "apts_p")
                ):
                    step_args = {"inputs": x, "labels": y}
                elif "asntr" in self.optimizer.__class__.__name__.lower():
                    x_d, y_d = self.sample_control_batch(1)
                    step_args = {
                        "inputs": x,
                        "labels": y,
                        "inputs_d": x_d,
                        "labels_d": y_d,
                    }
                else:
                    step_args = {"closure": general_closure}
                total_loss += self.optimizer.step(**step_args)

            # Print progress within the epoch
            self.epoch_progress = 100.0 * (batch_idx + 1) / len(self.train_loader)
            self.batch_dt = time.time() - batch_time_start
            self.iter_dt = self.batch_dt
            self.running_time = time.time() - self.total_start_time
            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1

        self.loss = total_loss / len(self.train_loader)
        self.adjust_batch_size(self.loss)
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
        self.total_start_time = time.time()
        self.epoch_num = 0
        self.epoch_time = time.time()
        for _ in range(self.config.epochs + 1):
            self.compute_epoch_loss()
            self.compute_accuracy()
            self.running_time = time.time() - self.total_start_time
            self.trigger_callbacks("on_epoch_end")
            self.epoch_num += 1

    def run_by_iter(self):
        model, cfg = self.model, self.config
        model.train()
        self.total_start_time = time.time()
        self.iter_time = time.time()
        self.data_iter = iter(self.train_loader)
        for self.iter_num in range(
            cfg.max_iters + 1 if cfg.max_iters is not None else float("inf")
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
                    data_chunks_amount=cfg.data_chunks_amount,
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
                    data_chunks_amount=cfg.data_chunks_amount,
                    compute_grad=True,
                    grad_norm_clip=cfg.grad_norm_clip,
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
                elif (
                    "apts_d" in self.optimizer.__class__.__name__.lower()
                    or "apts_p" in self.optimizer.__class__.__name__.lower()
                ):
                    self.loss += self.optimizer.step(inputs=x, labels=y)
                elif "asntr" in self.optimizer.__class__.__name__.lower():
                    x_d, y_d = self.sample_control_batch(1)
                    self.loss += self.optimizer.step(
                        inputs=x,
                        labels=y,
                        inputs_d=x_d,
                        labels_d=y_d,
                    )
                else:
                    self.loss = self.optimizer.step(closure=general_closure)

            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.running_time = tnow - self.total_start_time
            self.trigger_callbacks("on_batch_end")
            self.iter_time = tnow

    def compute_perplexity(self):
        pass
