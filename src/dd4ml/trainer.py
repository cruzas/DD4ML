"""
This code also runs with our dictionary-defined model, which is instantiated as a ParallelizedModel object.
Model handler takes care of the parallelized model logic. This is why this is slightly different from the trainer in mingpt.
"""

import inspect
import math
import numbers
import os
import time
from collections import defaultdict
from itertools import chain, count

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
from dd4ml.datasets.pinn_poisson import Poisson1DDataset
from dd4ml.utility import CfgNode as CN
from dd4ml.utility import closure, dprint

from .dataloaders import (
    GeneralizedDistributedDataLoader,
    MicroBatchFlattenSampler,
    MicroBatchOverlapSampler,
    OverlapBatchSampler,
)


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
        C.run_by_epoch = True  # if False, run by iteration instead of epochs, typically for transformer networks
        C.max_iters = 100000
        # optimizer
        C.learning_rate = 1e-3
        C.betas = (0.9, 0.999)  # for Adam
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = None
        # initial batch size and adaptive params
        C.batch_size = 128  # max batch size is length of dataset
        C.batch_inc_factor = 1.0  # factor to increase batch size
        C.loss_tol = 1e-2  # loss tolerance for adaptive batch size
        # APTS and TR
        C.delta = 0.1  # for trust region methods
        C.min_delta = 1e-3
        C.max_delta = 2.0
        C.data_parallel = True
        C.num_subdomains = 1  # for micro-batch splitting
        C.norm_type = 2  # for APTS_D (and possibly APTS_IP)
        C.glob_pass = False
        C.foc = False  # for APTS_D
        C.soc = False  # for APTS_D
        C.glob_dogleg = False  # for APTS_*, *TR
        C.loc_dogleg = False  # for APTS_*, *TR
        C.max_glob_iters = 1  # for APTS*
        C.max_loc_iters = 3  # for APTS*
        C.glob_second_order = False  # for APTS*
        C.loc_second_order = False  # for APTS*
        C.max_wolfe_iters = 5  # for APTS*
        C.max_zoom_iters = 5
        C.paper_tr_update = False  # for LSSR1_TR
        C.gradient_accumulation = True  # for APTS*
        C.accumulation_steps = 1  # for APTS*
        C.mem_length = 3  # for TR methods
        # For pipelining via pwm library
        C.data_chunks_amount = 1
        C.use_pmw = False
        C.loc_opt = None
        C.glob_opt = None
        C.overlap = 0.0
        C.adjust_batch_size_every_iters = 10000  # for use with LLMs and LSSR1_TR
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
            # print("Doing auto device selection...")
            self.device = (
                f"cuda:{torch.cuda.current_device()}"
                if dist.get_backend() != "gloo"
                else "cpu"
            )
        else:
            self.device = config.device

        self.model = self.model.to(self.device)
        # adaptive-batch state
        self.current_batch_size = config.batch_size
        self.max_batch_size_reached = False
        self.last_loss = float("inf")
        self.grad_evals = 0.0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

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

    def _get_num_batches(self):
        """Return the number of batches in the current train_loader."""
        return len(self.train_loader)

    def setup_data_loaders(self):
        """(Re)create train and test loaders; test loader always has zero overlap."""
        cfg, ds_train, ds_test = self.config, self.train_dataset, self.test_dataset
        # Ensure batch size does not exceed dataset
        if self.current_batch_size >= len(ds_train):
            print(f"Batch size {self.current_batch_size} >= dataset size {len(ds_train)}; "
                "using full dataset as single batch, overlap=0.")
            self.current_batch_size = len(ds_train)
            cfg.overlap = 0
        bs = self.current_batch_size

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # --- TRAIN LOADER ---
        if cfg.use_pmw:
            self.train_loader = GeneralizedDistributedDataLoader(
                model_handler=cfg.model_handler,
                dataset=ds_train,
                batch_size=bs,
                shuffle=False,
                overlap=cfg.overlap,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        else:
            shard_size = math.ceil(len(ds_train) / world_size)
            pp_bs = bs // world_size

            if pp_bs >= shard_size:
                print(f"Per-process batch size {pp_bs} >= shard size {shard_size}; "
                    "shard as single batch, overlap=0.")
                cfg.overlap = 0

            if pp_bs < 1:
                raise ValueError(f"Per-process batch size {pp_bs} < 1; increase global batch size.")

            base_train = DistributedSampler(
                ds_train, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
            )
            train_ov = OverlapBatchSampler(
                base_sampler=base_train,
                batch_size=pp_bs,
                overlap=cfg.overlap,
                drop_last=False,
            )

            num_sub = getattr(cfg, "num_subdomains", 1)
            if num_sub > 1:
                if cfg.overlap > 0:
                    train_micro = MicroBatchOverlapSampler(
                        overlap_sampler=train_ov,
                        num_subdomains=num_sub,
                        allow_empty_microbatches=False,
                    )
                    train_sampler = MicroBatchFlattenSampler(train_micro)
                    print(f"Using micro-batching with {num_sub} subdomains and overlap={cfg.overlap}")
                else:
                    print(f"Warning: num_subdomains={num_sub} requested but overlap=0. "
                        "Using regular batching.")
                    train_sampler = train_ov
            else:
                train_sampler = train_ov
                if cfg.overlap > 0:
                    print(f"Using overlap={cfg.overlap} between mini-batches (num_subdomains=1)")
                else:
                    print("Using regular batching (no overlap, num_subdomains=1)")

            self.train_loader = DataLoader(
                ds_train,
                batch_sampler=train_sampler,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            dprint(f"Number of batches in train_loader: {len(self.train_loader)}")

        # --- TEST LOADER (no overlap) ---
        if cfg.use_pmw:
            self.test_loader = DataLoader(
                ds_test,
                batch_size=bs,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        else:
            shard_size_test = math.ceil(len(ds_test) / world_size)
            pp_bs_test = bs // world_size

            if pp_bs_test >= shard_size_test:
                print(f"Per-process test batch size {pp_bs_test} >= shard size {shard_size_test}; "
                    "using full shard as single batch.")
            if pp_bs_test < 1:
                raise ValueError(f"Per-process test batch size {pp_bs_test} < 1; increase global batch size.")

            self.test_loader = DataLoader(
                ds_test,
                batch_size=pp_bs_test,
                sampler=DistributedSampler(
                    ds_test, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
                ),
                num_workers=cfg.num_workers,
                pin_memory=True,
            )


    def _asntr_present(self):
        # Check if ASNTR is the optimizer itself...
        asntr_is_opt = "asntr" in self.optimizer.__class__.__name__.lower()
        # ...or whether it is the global optimizer in the case of APTS*
        asntr_is_glob_opt = (
            hasattr(self.optimizer, "glob_opt")
            and "asntr" in self.optimizer.glob_opt.__class__.__name__.lower()
        )

        return asntr_is_glob_opt or asntr_is_opt

    def _lssr1_tr_present(self):
        # Check if LSSR1_TR is the optimizer itself...
        lssr1_tr_is_opt = "lssr1_tr" in self.optimizer.__class__.__name__.lower()
        # ...or whether it is the global optimizer in the case of APTS*
        lssr1_tr_is_glob_opt = (
            hasattr(self.optimizer, "glob_opt")
            and "lssr1_tr" in self.optimizer.glob_opt.__class__.__name__.lower()
        )
        return lssr1_tr_is_glob_opt or lssr1_tr_is_opt

    def _sgd_present(self):
        """Check if the optimizer is a standard SGD optimizer."""
        return "sgd" in self.optimizer.__class__.__name__.lower() or (
            hasattr(self.optimizer, "glob_opt")
            and "sgd" in self.optimizer.glob_opt.__class__.__name__.lower()
        )

    def _stay_here(self) -> bool:
        """Return True if either optimizer requests to keep the current batch,
        and reset both flags so they may be set again later."""
        stay = False

        if hasattr(self.optimizer, "move_to_next_batch"):
            stay |= not self.optimizer.move_to_next_batch
            self.optimizer.move_to_next_batch = True

        glob = getattr(self.optimizer, "glob_opt", None)
        if glob is not None and hasattr(glob, "move_to_next_batch"):
            stay |= not glob.move_to_next_batch
            glob.move_to_next_batch = True

        return stay

    def _inc_batch_size(self) -> bool:
        """Return True if either optimizer requests to increase the batch size,
        and reset both flags so they may be set again later."""
        inc = False

        if hasattr(self.optimizer, "inc_batch_size"):
            inc |= self.optimizer.inc_batch_size
            self.optimizer.inc_batch_size = False

        glob = getattr(self.optimizer, "glob_opt", None)
        if glob is not None and hasattr(glob, "inc_batch_size"):
            inc |= glob.inc_batch_size
            glob.inc_batch_size = False

        return inc

    def _adjust_batch_size(self, loss):
        """Increase batch size when current loss is greater than previous loss."""
        cfg = self.config
        if cfg.batch_inc_factor == 1:
            return  # no batch size adjustment

        # Check if the loss has increased and we have not reached the maximum batch size
        loss_inc_and_still_not_max_bs = (
            loss > self.last_loss - cfg.loss_tol and not self.max_batch_size_reached
        )

        # Condition that must be satisfied for LSSR1 to increase batch size
        lssr1_tr_cond = self._lssr1_tr_present() and loss_inc_and_still_not_max_bs

        # Condition that must be satisfied for ASNTR to increase batch size
        asntr_cond = (
            self._asntr_present()
            and self._inc_batch_size()
            and not self.max_batch_size_reached
        )

        # Condition that must be satisfied for SGD to increase batch size
        sgd_cond = self._sgd_present() and loss_inc_and_still_not_max_bs

        if lssr1_tr_cond or asntr_cond or sgd_cond:
            new_bs = min(
                int(math.floor(self.current_batch_size * cfg.batch_inc_factor)),
                len(self.train_dataset),
            )
            if new_bs == len(self.train_dataset):
                self.max_batch_size_reached = True
                dprint(
                    f"Batch size is already at maximum ({new_bs}). No more increases allowed."
                )
                if self.current_batch_size != new_bs:
                    self.current_batch_size = new_bs
                    self.setup_data_loaders()
                return

            dprint(f"Increasing batch size from {self.current_batch_size} to {new_bs}.")

            self.current_batch_size = new_bs

            # Rebuild data loaders to reflect the new batch size
            self.setup_data_loaders()

            # Invalidate cached shapes for any WeightParallelizedTensor parameters
            for p in self.model.parameters():
                if isinstance(p, WeightParallelizedTensor):
                    p.invalidate_shape_cache()
        self.last_loss = loss

    def _compute_hNk(self):
        """Compute the hNk value for the current batch size."""
        if self._asntr_present():
            # hNk = (N - k) / N, where N is the total number of samples and k is the current batch size
            return (len(self.train_dataset) - self.current_batch_size) / len(
                self.train_dataset
            )
        return None

    def sample_control_batch(self, batch_size: int = 1):
        """Randomly sample a small control batch from the training dataset."""
        idx = torch.randint(0, len(self.train_dataset), (batch_size,))
        xs, ys = zip(*(self.train_dataset[int(i)] for i in idx))
        xs = torch.stack([torch.as_tensor(x) for x in xs])
        ys = torch.as_tensor(ys)
        return xs.to(self.device), ys.to(self.device)

    def run(self):
        self.setup_data_loaders()
        if isinstance(self.train_dataset, Poisson1DDataset):
            self.run_by_epoch_PINN()
        elif self.config.run_by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()

    @torch.no_grad()
    def _eval_full_objective(self) -> float:
        """
        Evaluate f(w) = (1/n) Σ_i f_i(w) over the *entire* training set
        without disturbing the main train_loader iterator.

        Works for both cfg.use_pmw = {False, True}.
        """
        was_training = self.model.training
        self.model.eval()

        cfg = self.config
        if cfg.use_pmw:
            # same class as the training loader
            eval_loader = GeneralizedDistributedDataLoader(
                model_handler=cfg.model_handler,
                dataset=self.train_dataset,
                batch_size=self.current_batch_size,
                shuffle=False,
                overlap=0,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
        else:
            eval_loader = DataLoader(
                self.train_dataset,
                batch_size=self.current_batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

        total, n = 0.0, 0
        crit = self.criterion

        cond_std = not hasattr(self.model, "model_handler")
        cond_d_a = (
            hasattr(self.model, "model_handler")
            and self.model.model_handler.is_last_stage()
        )
        cond_d_b = False
        if not cond_std:  # identify the global rank that owns the last stage
            last_stage_rank = self.model.model_handler.get_stage_ranks(
                stage_name="last", mode="global"
            )[0]
            cond_d_b = dist.get_rank() == last_stage_rank

        for x, y in eval_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            if not cond_std:  # pipeline -> extract the shard
                out = out[0]
            if cond_std or (cond_d_a and cond_d_b):
                loss = crit(out, y).item()
                bs = y.size(0)
                total += loss * bs
                n += bs

        if dist.is_initialized():
            buf = torch.tensor([total, n], device=self.device, dtype=torch.float32)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            total, n = buf.tolist()

        if was_training:
            self.model.train()
        return total / n

    def _tr_or_apts_present(self):
        """Check if the optimizer is a trust-region or APTS optimizer."""
        return (
            "tr" in self.optimizer.__class__.__name__.lower()
            or "apts" in self.optimizer.__class__.__name__.lower()
        )

    def compute_accuracy(self):
        """Compute test accuracy across all processes (and pipeline stages)."""
        if isinstance(self.test_dataset, Poisson1DDataset):
            self.accuracy = float("nan")
            return

        model = self.model
        model.eval()

        # local counters
        correct = torch.tensor(0, device=self.device)
        total = torch.tensor(0, device=self.device)

        # pipeline conditions
        cond_std = not hasattr(model, "model_handler")
        cond_d_a = (
            hasattr(model, "model_handler") and model.model_handler.is_last_stage()
        )
        cond_d_b = False
        if not cond_std:
            last_rank = model.model_handler.get_stage_ranks(
                stage_name="last", mode="global"
            )[0]
            cond_d_b = dist.get_rank() == last_rank

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                if not cond_std:
                    outputs = outputs[0]
                if cond_std or (cond_d_a and cond_d_b):
                    _, preds = outputs.max(dim=1)
                    correct += (preds == y).sum()
                    total += y.size(0)

        # aggregate across all GPUs/ranks
        if dist.is_initialized():
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total,   op=dist.ReduceOp.SUM)

        self.accuracy = 100.0 * correct.item() / total.item()
        model.train()


    @torch.no_grad()
    def compute_current_train_perplexity(self):
        """Compute train perplexity directly from the latest batch loss."""
        # self.loss may be a float or a zero‐dim tensor
        loss_val = self.loss.item() if torch.is_tensor(self.loss) else float(self.loss)
        return math.exp(loss_val)

    @torch.no_grad()
    def compute_test_perplexity(self):
        """Compute the model's perplexity on the test set (supports distributed/pipeline parallel)."""
        model = self.model
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        # Determine when to accumulate (last pipeline stage or standard model)
        cond_std = not hasattr(model, "model_handler")
        cond_d_a = (
            hasattr(model, "model_handler") and model.model_handler.is_last_stage()
        )
        cond_d_b = False
        if not cond_std:
            last_stage_rank = model.model_handler.get_stage_ranks(
                stage_name="last", mode="global"
            )[0]
            cond_d_b = dist.get_rank() == last_stage_rank

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            outputs = model(x)
            if not cond_std:
                outputs = outputs[0]

            if cond_std or (cond_d_a and cond_d_b):
                # Flatten logits and targets
                logits = outputs.view(-1, outputs.size(-1))
                targets = y.view(-1)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()

        # Aggregate across all ranks
        if dist.is_initialized():
            buf = torch.tensor(
                [total_loss, total_tokens], device=self.device, dtype=torch.float32
            )
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = buf.tolist()

        model.train()
        return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    def _train_one_batch(self, x, y, first_grad: bool):
        """Runs forward/backward/step on a single (x,y). Returns (batch_loss, batch_grad, bs)."""
        x, y = x.to(self.device), y.to(self.device)
        bs = y.size(0)

        # Special handling for Poisson1D PINN dataset
        if isinstance(self.train_dataset, Poisson1DDataset):
            return self._train_one_batch_PINN(x, y, first_grad)

        # optional control batch for ASNTR
        if self._asntr_present():
            x_d, y_d = self.sample_control_batch(1)
        else:
            x_d = y_d = None

        # warm-up closure if needed
        if first_grad:
            c = closure(
                x,
                y,
                self.criterion,
                self.model,
                data_chunks_amount=self.config.data_chunks_amount,
                compute_grad=False,
            )
            batch_loss = c()
            batch_grad = None
        else:
            # full-gradient closure
            general = closure(
                x,
                y,
                criterion=self.criterion,
                model=self.model,
                data_chunks_amount=self.config.data_chunks_amount,
                compute_grad=True,
                grad_norm_clip=self.config.grad_norm_clip,
            )
            
            sig = inspect.signature(self.optimizer.step).parameters
            step_args = {}
            if "final_subdomain_closure" in sig:

                def final_subdomain_closure(outputs, y=y):
                    ys = y.chunk(len(outputs))
                    return [self.criterion(o, yc) for o, yc in zip(outputs, ys)]

                step_args = {
                    "closure": general,
                    "final_subdomain_closure": final_subdomain_closure,
                }
            elif any(
                k in self.optimizer.__class__.__name__.lower()
                for k in ("apts_d", "apts_p")
            ):
                step_args = {
                    "inputs": x,
                    "labels": y,
                    "inputs_d": x_d,
                    "labels_d": y_d,
                    "hNk": self._compute_hNk(),
                }
            elif "asntr" in self.optimizer.__class__.__name__.lower():
                closure_d = closure(
                    x_d,
                    y_d,
                    criterion=self.criterion,
                    model=self.model,
                    data_chunks_amount=1,
                    compute_grad=True,
                    grad_norm_clip=self.config.grad_norm_clip,
                )
                step_args = {
                    "closure_main": general,
                    "closure_d": closure_d,
                    "hNk": self._compute_hNk(),
                }
            else:
                step_args = {"closure": general}

            result = self.optimizer.step(**step_args)
            # self.grad_evals += getattr(self.optimizer, "grad_evals", 0) / len(
            #     self.train_loader
            # )
            if hasattr(self.optimizer, "grad_evals"):
                self.grad_evals += self.optimizer.grad_evals * (bs * self.world_size) / len(self.train_dataset)
            else:
                self.grad_evals += 1 * (bs * self.world_size) / len(self.train_dataset)

            if isinstance(result, numbers.Number) or (
                torch.is_tensor(result) and result.ndim == 0
            ):
                batch_loss = float(result)
                batch_grad = None
            else:
                batch_loss, batch_grad = result

        return batch_loss, batch_grad, bs

    def _train_one_batch_PINN(self, x, y, first_grad: bool):
        """Specialized training step for the Poisson PINN dataset."""
        x, y = x.to(self.device), y.to(self.device)
        x.requires_grad_(True)
        bs = y.size(0)
        if hasattr(self.criterion, "current_x"):
            self.criterion.current_x = x

        x_d = y_d = None
        if self._asntr_present():
            x_d, y_d = self.sample_control_batch(1)

        general = closure(
            x,
            y,
            criterion=self.criterion,
            model=self.model,
            data_chunks_amount=self.config.data_chunks_amount,
            compute_grad=True,
            grad_norm_clip=self.config.grad_norm_clip,
        )

        sig = inspect.signature(self.optimizer.step).parameters
        if "final_subdomain_closure" in sig:

            def final_subdomain_closure(outputs, y=y):
                ys = y.chunk(len(outputs))
                return [self.criterion(o, yc) for o, yc in zip(outputs, ys)]

            step_args = {
                "closure": general,
                "final_subdomain_closure": final_subdomain_closure,
            }
        elif any(
            k in self.optimizer.__class__.__name__.lower() for k in ("apts_d", "apts_p")
        ):
            step_args = {
                "inputs": x,
                "labels": y,
                "inputs_d": x_d,
                "labels_d": y_d,
                "hNk": self._compute_hNk(),
            }
        elif "asntr" in self.optimizer.__class__.__name__.lower():
            closure_d = closure(
                x_d,
                y_d,
                criterion=self.criterion,
                model=self.model,
                data_chunks_amount=1,
                compute_grad=True,
                grad_norm_clip=self.config.grad_norm_clip,
            )
            step_args = {
                "closure_main": general,
                "closure_d": closure_d,
                "hNk": self._compute_hNk(),
            }
        else:
            step_args = {"closure": general}

        result = self.optimizer.step(**step_args)
        if hasattr(self.optimizer, "grad_evals"):
            self.grad_evals += self.optimizer.grad_evals * (bs * self.world_size) / len(self.train_dataset)
        else:
            self.grad_evals += 1 * (bs * self.world_size) / len(self.train_dataset)

        if isinstance(result, numbers.Number) or (torch.is_tensor(result) and result.ndim == 0):
            batch_loss = float(result)
            batch_grad = None
        else:
            batch_loss, batch_grad = result

        return batch_loss, batch_grad, bs

    def run_by_epoch(self):
        """Run the training loop by epochs with correct global loss accumulation."""
        # each rank processes N/world_size samples per epoch
        self.num_training_samples_per_process = len(self.train_dataset) / self.world_size
        self.total_start_time = time.time()
        self.epoch_time = time.time()
        self.epoch_num = 0

        dprint(f"Total number of training samples per process: {self.num_training_samples_per_process}")

        while self.epoch_num <= self.config.epochs:
            epoch_loss = 0.0
            total_samples = 0
            it = iter(self.train_loader)
            first = (self.epoch_num == 0)
            
            if hasattr(self.train_loader.sampler, "set_epoch"):
                # Set the epoch for the sampler to shuffle data correctly
                self.train_loader.sampler.set_epoch(self.epoch_num)
            
            # loop until this rank has seen its shard (plus any overlap)
            while total_samples < self.num_training_samples_per_process:
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    x, y = next(it)

                batch_loss, batch_grad, bs = self._train_one_batch(x, y, first)
                total_samples += bs

                # weight by global sample count
                epoch_loss += batch_loss * (bs * self.world_size / len(self.train_dataset))
                self.loss = epoch_loss

                stay = self._stay_here()
                if stay:
                    it = chain([(x, y)], it)
                elif self._asntr_present():
                    old_n = len(self.train_loader)
                    self._adjust_batch_size(self.loss)
                    if len(self.train_loader) != old_n:
                        it = iter(self.train_loader)

            # optional full‐objective adjustment
            if (self._lssr1_tr_present() or self._sgd_present()) and self.epoch_num % 3 == 0:
                full = self._eval_full_objective()
                self._adjust_batch_size(full)

            self.compute_accuracy()
            self.epoch_dt = time.time() - self.epoch_time
            self.running_time = time.time() - self.total_start_time
            self.epoch_time = time.time()
            self.trigger_callbacks("on_epoch_end")
            self.epoch_num += 1
        

    def run_by_epoch_PINN(self):
        """Simplified epoch loop for PINN datasets."""
        self.num_training_samples_per_process = len(self.train_dataset) / self.world_size
        self.total_start_time = time.time()
        self.epoch_time = time.time()
        self.epoch_num = 0

        while self.epoch_num <= self.config.epochs:
            epoch_loss = 0.0
            total_samples = 0
            it = iter(self.train_loader)

            while total_samples < self.num_training_samples_per_process:
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    x, y = next(it)

                batch_loss, batch_grad, bs = self._train_one_batch_PINN(x, y, False)
                total_samples += bs
                epoch_loss += batch_loss * (bs * self.world_size / len(self.train_dataset))
                self.loss = epoch_loss

            self.accuracy = float("nan")  # PINN datasets do not have accuracy
            self.epoch_dt = time.time() - self.epoch_time
            self.running_time = time.time() - self.total_start_time
            self.epoch_time = time.time()
            self.trigger_callbacks("on_epoch_end")
            self.epoch_num += 1


    def run_by_iter(self):
        cfg = self.config
        cfg.adjust_batch_size_every_iters = int(cfg.adjust_batch_size_every_iters)
        self.total_start_time = time.time()
        self.iter_time = time.time()
        it = iter(self.train_loader)
        stay = False

        for self.iter_num in range(cfg.max_iters + 1):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                x, y = next(it)

            first = self.iter_num == 0
            batch_loss, batch_grad, bs = self._train_one_batch(x, y, first)
            self.loss = batch_loss

            stay = self._stay_here()
            if stay:
                it = chain([(x, y)], it)

            if self._asntr_present():
                self._adjust_batch_size(self.loss)
            elif (
                (self._lssr1_tr_present() or self._sgd_present())
                and self.iter_num > 0
                and self.iter_num % cfg.adjust_batch_size_every_iters == 0
            ):
                full = self._eval_full_objective()
                self._adjust_batch_size(full)

            self.curr_train_perplexity = self.compute_current_train_perplexity()

            self.iter_dt = time.time() - self.iter_time
            self.running_time = time.time() - self.total_start_time
            self.trigger_callbacks("on_batch_end")
            self.iter_time = time.time()
