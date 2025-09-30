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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from dd4ml.datasets.deeponet_sine import SineOperatorDataset
from dd4ml.datasets.pinn_allencahn import AllenCahn1DDataset
from dd4ml.datasets.pinn_allencahn_time import AllenCahn1DTimeDataset
from dd4ml.datasets.pinn_poisson import Poisson1DDataset
from dd4ml.datasets.pinn_poisson2d import Poisson2DDataset
from dd4ml.datasets.pinn_poisson3d import Poisson3DDataset

# Cache dataset type tuples for efficiency
PINN_DATASETS = (
    Poisson1DDataset,
    Poisson2DDataset,
    Poisson3DDataset,
    AllenCahn1DDataset,
    AllenCahn1DTimeDataset,
)

ACCURACY_EXCLUDED_DATASETS = (
    Poisson1DDataset,
    Poisson2DDataset,
    Poisson3DDataset,
    AllenCahn1DDataset,
    SineOperatorDataset,
)
from dd4ml.pmw.weight_parallelized_tensor import WeightParallelizedTensor
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
        # data shuffling
        C.shuffle = True
        # random seed for reproducibility
        C.seed = 42
        # floating point precision for distributed training
        C.precision = "float32"  # Options: "float32", "float64"
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
        C.contiguous_subdomains = False
        C.exclusive = True
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

        # Set up precision handling FIRST
        self.precision_dtype = torch.float32 if config.precision == "float32" else torch.float64

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
        # Set model precision to match training precision
        if self.precision_dtype == torch.float64:
            self.model = self.model.double()
            # Ensure optimizer uses double precision parameters
            if hasattr(self.optimizer, 'param_groups'):
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.dtype != torch.float64:
                            p.data = p.data.double()
        else:
            self.model = self.model.float()
        # adaptive-batch state
        self.current_batch_size = config.batch_size
        self.max_batch_size_reached = False
        self.last_loss = float("inf")
        self.grad_evals = 0.0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Cache optimizer type checks for performance
        optimizer_name = self.optimizer.__class__.__name__.lower()
        self._is_tr_or_apts = "tr" in optimizer_name or "apts" in optimizer_name
        self._is_apts_ip = "apts_ip" in optimizer_name
        self._is_apts_pinn = "apts_pinn" in optimizer_name
        self._is_asntr = "asntr" in optimizer_name
        self._is_lssr1_tr = "lssr1_tr" in optimizer_name

        # Set up precision handling
        self.precision_dtype = torch.float32 if config.precision == "float32" else torch.float64

        # Pre-computed efficiency constants
        self._dataset_size_inv = 1.0 / len(self.train_dataset)
        if not self._apts_ip_present():
            self._loss_scaling_factor = self.world_size * self._dataset_size_inv
        else:
            self._loss_scaling_factor = self._dataset_size_inv

        # Pre-allocated buffers for metric computation (guaranteed safe in @torch.no_grad())
        self._metric_buffer = torch.zeros(2, device=self.device, dtype=torch.float32)
        self._perplexity_buffer = torch.zeros(
            2, device=self.device, dtype=torch.float32
        )

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

    def _update_loss_scaling_factor(self):
        """Update loss scaling factor when dataset or batch configuration changes."""
        self._dataset_size_inv = 1.0 / len(self.train_dataset)
        if not self._apts_ip_present():
            self._loss_scaling_factor = self.world_size * self._dataset_size_inv
        else:
            self._loss_scaling_factor = self._dataset_size_inv

    def setup_data_loaders(self):
        """(Re)create train and test loaders; test loader always has zero overlap."""
        # Update loss scaling factor in case dataset or configuration changed
        self._update_loss_scaling_factor()

        cfg, ds_train, ds_test = self.config, self.train_dataset, self.test_dataset
        # Ensure batch size does not exceed dataset
        if self.current_batch_size >= len(ds_train):
            print(
                f"Batch size {self.current_batch_size} >= dataset size {len(ds_train)}; "
                "using full dataset as single batch, overlap=0."
            )
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
            if cfg.contiguous_subdomains:
                dataset_len = len(ds_train)
                pp_bs = bs
                if pp_bs >= dataset_len:
                    print(
                        f"Batch size {pp_bs} >= dataset size {dataset_len}; "
                        "using full dataset as single batch, overlap=0."
                    )
                    cfg.overlap = 0
                if pp_bs < 1:
                    raise ValueError(f"Batch size {pp_bs} < 1; increase batch size.")
                shuffle_flag = cfg.shuffle and not isinstance(ds_train, AllenCahn1DDataset)
                if shuffle_flag:
                    base_train = RandomSampler(ds_train)
                else:
                    base_train = SequentialSampler(ds_train)
                train_ov = OverlapBatchSampler(
                    base_sampler=base_train,
                    batch_size=pp_bs,
                    overlap=cfg.overlap,
                    drop_last=self._apts_ip_present(),
                )
                train_sampler = train_ov
                if cfg.overlap > 0:
                    print(
                        f"Using overlap={cfg.overlap} between mini-batches (contiguous subdomains)"
                    )
                else:
                    print("Using regular batching with contiguous subdomains")
            else:
                shard_size = math.ceil(len(ds_train) / world_size)
                pp_bs = bs // world_size

                if pp_bs >= shard_size:
                    print(
                        f"Per-process batch size {pp_bs} >= shard size {shard_size}; "
                        "shard as single batch, overlap=0."
                    )
                    cfg.overlap = 0

                if pp_bs < 1:
                    raise ValueError(
                        f"Per-process batch size {pp_bs} < 1; increase global batch size."
                    )

                shuffle_flag = cfg.shuffle and not isinstance(ds_train, AllenCahn1DDataset)

                # For deterministic distributed SGD, we want identical data ordering
                # across different numbers of processes
                if shuffle_flag:
                    base_train = DistributedSampler(
                        ds_train,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=True,
                        drop_last=self._apts_ip_present(),
                        seed=getattr(cfg, 'seed', 42),
                    )
                else:
                    # For deterministic distributed SGD: ensure identical batch composition
                    # regardless of number of processes by using consistent global batching

                    # Create a custom sampler that ensures all processes see the same logical batches
                    # but distributed across processes
                    class DeterministicDistributedSampler:
                        def __init__(self, dataset, num_replicas, rank, batch_size):
                            self.dataset_size = len(dataset)
                            self.num_replicas = num_replicas
                            self.rank = rank
                            self.batch_size = batch_size

                            # Calculate how many complete global batches we can make
                            self.num_global_batches = self.dataset_size // batch_size

                        def __iter__(self):
                            indices = []
                            # For each global batch, distribute samples across processes
                            for global_batch_idx in range(self.num_global_batches):
                                batch_start = global_batch_idx * self.batch_size
                                batch_end = batch_start + self.batch_size
                                global_batch = list(range(batch_start, batch_end))

                                # Distribute this global batch across processes
                                per_process_size = len(global_batch) // self.num_replicas
                                start_idx = self.rank * per_process_size
                                end_idx = start_idx + per_process_size

                                indices.extend(global_batch[start_idx:end_idx])

                            return iter(indices)

                        def __len__(self):
                            per_process_size = self.batch_size // self.num_replicas
                            return self.num_global_batches * per_process_size

                    base_train = DeterministicDistributedSampler(
                        ds_train, world_size, rank, self.current_batch_size
                    )
                train_ov = OverlapBatchSampler(
                    base_sampler=base_train,
                    batch_size=pp_bs,
                    overlap=cfg.overlap,
                    drop_last=self._apts_ip_present(),
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
                        print(
                            f"Using micro-batching with {num_sub} subdomains and overlap={cfg.overlap}"
                        )
                    else:
                        print(
                            f"Warning: num_subdomains={num_sub} requested but overlap=0. "
                            "Using regular batching."
                        )
                        train_sampler = train_ov
                else:
                    train_sampler = train_ov
                    if cfg.overlap > 0:
                        print(
                            f"Using overlap={cfg.overlap} between mini-batches (num_subdomains=1)"
                        )
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
            if cfg.contiguous_subdomains:
                shard_size_test = len(ds_test)
            else:
                shard_size_test = math.ceil(len(ds_test) / world_size)
            pp_bs_test = bs // world_size

            if pp_bs_test >= shard_size_test:
                print(
                    f"Per-process test batch size {pp_bs_test} >= shard size {shard_size_test}; "
                    "using full shard as single batch."
                )
            if pp_bs_test < 1:
                raise ValueError(
                    f"Per-process test batch size {pp_bs_test} < 1; increase global batch size."
                )

            if cfg.contiguous_subdomains:
                self.test_loader = DataLoader(
                    ds_test,
                    batch_size=pp_bs_test,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                )
            else:
                self.test_loader = DataLoader(
                    ds_test,
                    batch_size=pp_bs_test,
                    sampler=DistributedSampler(
                        ds_test,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=False,
                        drop_last=self._apts_ip_present(),
                        seed=getattr(cfg, 'seed', 42),
                    ),
                    num_workers=cfg.num_workers,
                    pin_memory=True,
                )

    def _apts_ip_present(self):
        """Check if APTS_IP is the optimizer itself..."""
        return self._is_apts_ip

    def _apts_pinn_present(self):
        """Check if APTS_PINN is the optimizer itself..."""
        return self._is_apts_pinn

    def _asntr_present(self):
        # Check if ASNTR is the optimizer itself...
        asntr_is_opt = self._is_asntr
        # ...or whether it is the global optimizer in the case of APTS*
        asntr_is_glob_opt = (
            hasattr(self.optimizer, "glob_opt")
            and "asntr" in self.optimizer.glob_opt.__class__.__name__.lower()
        )

        return asntr_is_glob_opt or asntr_is_opt

    def _lssr1_tr_present(self):
        # Check if LSSR1_TR is the optimizer itself...
        lssr1_tr_is_opt = self._is_lssr1_tr
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
        samples = [self.train_dataset[int(i)] for i in idx]
        first = samples[0]
        if isinstance(first, (tuple, list)):
            if len(first) == 2:
                xs, ys = zip(*samples)
                if isinstance(xs[0], (tuple, list)):
                    branch, trunk = zip(*xs)
                    branch = torch.stack([torch.as_tensor(b) for b in branch])
                    trunk = torch.stack([torch.as_tensor(t) for t in trunk])
                    xs = (branch, trunk)
                else:
                    xs = torch.stack([torch.as_tensor(x) for x in xs])
                ys = torch.stack([torch.as_tensor(y) for y in ys])
            elif len(first) == 3:
                xs, ts, ys = zip(*samples)
                xs = torch.stack([torch.as_tensor(x) for x in xs])
                ts = torch.stack([torch.as_tensor(t) for t in ts])
                xs = torch.cat([xs, ts], dim=1)
                ys = torch.stack([torch.as_tensor(y) for y in ys])
            else:
                raise ValueError("Unsupported sample format for control batch")
        else:
            xs = torch.stack([torch.as_tensor(s) for s in samples])
            ys = None
        return self._move_to_device(xs), self._move_to_device(ys)

    def run(self):
        self.setup_data_loaders()
        if isinstance(self.train_dataset, PINN_DATASETS):
            self.run_by_epoch_PINN()
        elif self.config.run_by_epoch:
            self.run_by_epoch()
        else:
            self.run_by_iter()

    def _eval_full_objective(self) -> float:
        """
        Evaluate f(w) = (1/n) Σ_i f_i(w) over the *entire* training set
        without disturbing the main train_loader iterator.

        Works for both cfg.use_pmw = {False, True}.
        """
        was_training = self.model.training
        self.model.eval()

        requires_grad_eval = isinstance(
            self.train_dataset,
            (
                Poisson1DDataset,
                Poisson2DDataset,
                Poisson3DDataset,
                AllenCahn1DDataset,
                AllenCahn1DTimeDataset,
            ),
        )
        context = torch.enable_grad if requires_grad_eval else torch.no_grad

        with context():
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

            for batch in eval_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, t, y = batch
                    x = torch.cat([x, t], dim=1)
                else:
                    x, y = batch
                x = self._move_to_device(x)
                y = self._move_to_device(y)

                # PINN datasets operate in double precision. During training we
                # explicitly cast inputs and targets to ``float64`` in
                # ``_train_one_batch_PINN`` and convert the model and loss in
                # ``run_by_epoch_PINN``.  However, the evaluation path used by
                # ``_eval_full_objective`` did not perform this cast, leading to
                # dtype mismatches between ``float32`` inputs and ``float64``
                # model weights when optimizers such as SGD were used.  Align
                # the evaluation dtype with the training dtype by promoting
                # PINN tensors to double here as well.
                if isinstance(
                    self.train_dataset,
                    (
                        Poisson1DDataset,
                        Poisson2DDataset,
                        Poisson3DDataset,
                        AllenCahn1DDataset,
                        AllenCahn1DTimeDataset,
                    ),
                ):
                    if isinstance(x, torch.Tensor):
                        x = x.double()
                    elif isinstance(x, (list, tuple)):
                        x = type(x)(
                            t.double() if isinstance(t, torch.Tensor) else t for t in x
                        )
                    y = y.double()

                if isinstance(x, torch.Tensor):
                    x.requires_grad_(True)
                elif isinstance(x, (list, tuple)):
                    for t in x:
                        if isinstance(t, torch.Tensor):
                            t.requires_grad_(True)
                out = self.model(x)
                if not cond_std:  # pipeline -> extract the shard
                    out = out[0]
                if cond_std or (cond_d_a and cond_d_b):
                    if hasattr(crit, "current_xy"):
                        crit.current_xy = x
                    if hasattr(crit, "current_x"):
                        crit.current_x = x
                    if hasattr(crit, "current_xyz"):
                        crit.current_xyz = x
                    if hasattr(crit, "current_xt"):
                        crit.current_xt = x
                    loss = crit(out, y.long()).item()
                    bs = y.size(0)
                    total += loss * bs
                    n += bs

        if dist.is_initialized():
            # Reuse pre-allocated buffer for efficiency
            self._metric_buffer[0] = total
            self._metric_buffer[1] = n
            dist.all_reduce(self._metric_buffer, op=dist.ReduceOp.SUM)
            total, n = self._metric_buffer.tolist()

        if was_training:
            self.model.train()
        return total / n

    def _tr_or_apts_present(self):
        """Check if the optimizer is a trust-region or APTS optimizer."""
        return self._is_tr_or_apts

    def _move_to_device(self, data):
        if isinstance(data, (list, tuple)):
            return type(data)(d.to(self.device, dtype=self.precision_dtype) for d in data)
        return data.to(self.device, dtype=self.precision_dtype)

    def compute_accuracy(self):
        """Compute test accuracy across all processes (and pipeline stages)."""
        if isinstance(self.test_dataset, ACCURACY_EXCLUDED_DATASETS):
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
                x = self._move_to_device(x)
                y = self._move_to_device(y)
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
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

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
            x = self._move_to_device(x)
            y = self._move_to_device(y)
            outputs = model(x)
            if not cond_std:
                outputs = outputs[0]

            if cond_std or (cond_d_a and cond_d_b):
                # Flatten logits and targets
                logits = outputs.view(-1, outputs.size(-1))
                targets = y.view(-1)
                loss = self.criterion(logits, targets.long())
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()

        # Aggregate across all ranks using pre-allocated buffer
        if dist.is_initialized():
            self._perplexity_buffer[0] = total_loss
            self._perplexity_buffer[1] = total_tokens
            dist.all_reduce(self._perplexity_buffer, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = self._perplexity_buffer.tolist()

        model.train()
        return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    def _train_one_batch(self, x, y, first_grad: bool):
        """Runs forward/backward/step on a single (x,y). Returns (batch_loss, batch_grad, bs)."""
        x = self._move_to_device(x)
        y = self._move_to_device(y)
        bs = y.size(0)

        # Special handling for PINN datasets
        if isinstance(
            self.train_dataset,
            (
                Poisson1DDataset,
                Poisson2DDataset,
                Poisson3DDataset,
                AllenCahn1DDataset,
                AllenCahn1DTimeDataset,
            ),
        ):
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
                precision_dtype=self.precision_dtype,
            )
            # Add barrier to ensure all processes are synchronized before calling closure
            if dist.is_initialized():
                dist.barrier()
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
                precision_dtype=self.precision_dtype,
            )

            sig = inspect.signature(self.optimizer.step).parameters
            step_args = {}
            if "final_subdomain_closure" in sig:

                def final_subdomain_closure(outputs, y=y):
                    ys = y.chunk(len(outputs))
                    return [self.criterion(o, yc.long()) for o, yc in zip(outputs, ys)]

                step_args = {
                    "closure": general,
                    "final_subdomain_closure": final_subdomain_closure,
                }
            elif any(
                k in self.optimizer.__class__.__name__.lower()
                for k in ("apts_d", "apts_p", "apts_pinn")
            ):
                step_args = {
                    "inputs": x,
                    "labels": y,
                    "inputs_d": x_d,
                    "labels_d": y_d,
                    "hNk": self._compute_hNk(),
                }
            elif self._is_asntr:
                closure_d = closure(
                    x_d,
                    y_d,
                    criterion=self.criterion,
                    model=self.model,
                    data_chunks_amount=1,
                    compute_grad=True,
                    grad_norm_clip=self.config.grad_norm_clip,
                    precision_dtype=self.precision_dtype,
                )
                step_args = {
                    "closure_main": general,
                    "closure_d": closure_d,
                    "hNk": self._compute_hNk(),
                }
            else:
                step_args = {"closure": general}

            result = self.optimizer.step(**step_args)

            # Debug: verify optimizer result consistency (disabled)
            # if hasattr(self, 'epoch_num') and self.epoch_num <= 2:
            #     if dist.is_initialized():
            #         print(f"E{self.epoch_num} P{dist.get_rank()}: optimizer_result={result:.6f}")
            #     else:
            #         print(f"E{self.epoch_num} P0: optimizer_result={result:.6f}")

            # self.grad_evals += getattr(self.optimizer, "grad_evals", 0) / len(
            #     self.train_loader
            # )
            if not self._apts_ip_present():
                if hasattr(self.optimizer, "grad_evals"):
                    self.grad_evals += (
                        self.optimizer.grad_evals
                        * (bs * self.world_size)
                        / len(self.train_dataset)
                    )
                else:
                    self.grad_evals += (
                        1 * (bs * self.world_size) / len(self.train_dataset)
                    )
            else:
                if hasattr(self.optimizer, "grad_evals"):
                    self.grad_evals += (
                        self.optimizer.grad_evals * bs / len(self.train_dataset)
                    )
                else:
                    self.grad_evals += 1 * bs / len(self.train_dataset)

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
        x = self._move_to_device(x).double()
        y = self._move_to_device(y).double()
        x.requires_grad_(True)
        if isinstance(self.criterion, Poisson3DDataset):
            y.requires_grad_(True)

        bs = y.size(0)
        # update criterion’s current points
        if hasattr(self.criterion, "current_xy"):
            self.criterion.current_xy = x
        if hasattr(self.criterion, "current_x"):
            self.criterion.current_x = x
        if hasattr(self.criterion, "current_xyz"):
            self.criterion.current_xyz = x
        if hasattr(self.criterion, "current_xt"):
            self.criterion.current_xt = x

        # control batch for ASNTR
        x_d = y_d = None
        if self._asntr_present():
            x_d, y_d = self.sample_control_batch(1)

        # warm-up pass: compute loss only
        if first_grad:
            warmup = closure(
                x,
                y,
                criterion=self.criterion,
                model=self.model,
                data_chunks_amount=self.config.data_chunks_amount,
                compute_grad=True,
                precision_dtype=self.precision_dtype,
            )
            batch_loss = warmup()
            batch_grad = None
            return batch_loss, batch_grad, bs

        # full-gradient pass and optimizer step
        general = closure(
            x,
            y,
            criterion=self.criterion,
            model=self.model,
            data_chunks_amount=self.config.data_chunks_amount,
            compute_grad=True,
            grad_norm_clip=self.config.grad_norm_clip,
            precision_dtype=self.precision_dtype,
        )

        sig = inspect.signature(self.optimizer.step).parameters
        if "final_subdomain_closure" in sig:

            def final_subdomain_closure(outputs, y=y):
                ys = y.chunk(len(outputs))
                return [self.criterion(o, yc.long()) for o, yc in zip(outputs, ys)]

            step_args = {
                "closure": general,
                "final_subdomain_closure": final_subdomain_closure,
            }
        elif any(
            k in self.optimizer.__class__.__name__.lower()
            for k in ("apts_d", "apts_p", "apts_pinn")
        ):
            step_args = {
                "inputs": x,
                "labels": y,
                "inputs_d": x_d,
                "labels_d": y_d,
                "hNk": self._compute_hNk(),
            }
        elif self._is_asntr:
            closure_d = closure(
                x_d,
                y_d,
                criterion=self.criterion,
                model=self.model,
                data_chunks_amount=1,
                compute_grad=True,
                grad_norm_clip=self.config.grad_norm_clip,
                precision_dtype=self.precision_dtype,
            )
            step_args = {
                "closure_main": general,
                "closure_d": closure_d,
                "hNk": self._compute_hNk(),
            }
        else:
            step_args = {"closure": general}

        result = self.optimizer.step(**step_args)

        # track grad_evals
        if not self._apts_ip_present():
            nevals = (
                self.optimizer.grad_evals
                if hasattr(self.optimizer, "grad_evals")
                else 1
            )
            if not self._apts_pinn_present():
                # APTS_D or APTS_P
                self.grad_evals += (
                    nevals * (bs * self.world_size) / len(self.train_dataset)
                )
            else:
                # APTS_PINN
                self.grad_evals += nevals * bs / len(self.train_dataset)
        else:
            nevals = (
                self.optimizer.grad_evals
                if hasattr(self.optimizer, "grad_evals")
                else 1
            )
            self.grad_evals += nevals * bs / len(self.train_dataset)

        # unpack result
        if isinstance(result, numbers.Number) or (
            torch.is_tensor(result) and result.ndim == 0
        ):
            batch_loss = float(result)
            batch_grad = None
        else:
            batch_loss, batch_grad = result

        return batch_loss, batch_grad, bs

    def run_by_epoch(self):
        """Run the training loop by epochs with correct global loss accumulation."""
        # Ensure model and optimizer precision consistency for float64 training
        if self.precision_dtype == torch.float64:
            self.model = self.model.double()
            if hasattr(self.optimizer, "loc_model") and self.optimizer.loc_model is not None:
                # APTS-based optimizers keep a separate local model that
                # is cloned during initialisation.  When switching the main
                # model to ``float64`` we must also convert this local copy,
                # otherwise forward passes inside the optimizer will mix
                # ``float32`` weights with ``float64`` inputs, triggering
                # dtype mismatch errors.
                self.optimizer.loc_model = self.optimizer.loc_model.double()

        # each rank processes N/world_size samples per epoch
        self.total_start_time = time.time()
        self.epoch_time = time.time()
        self.epoch_num = 0

        if not self._apts_ip_present():
            self.num_training_samples_per_process = (
                len(self.train_dataset) / self.world_size
            )
        else:
            self.num_training_samples_per_process = len(self.train_dataset)
        dprint(
            f"Total number of training samples per process: {self.num_training_samples_per_process}"
        )

        while self.epoch_num <= self.config.epochs:
            epoch_loss = 0.0
            total_samples = 0
            it = iter(self.train_loader)
            first = self.epoch_num == 0

            if hasattr(self.train_loader.sampler, "set_epoch"):
                # Set the epoch for the sampler to shuffle data correctly
                self.train_loader.sampler.set_epoch(self.epoch_num)

            # loop until this rank has seen its shard (plus any overlap)
            while total_samples < self.num_training_samples_per_process:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    batch = next(it)

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, t, y = batch
                    x = torch.cat([x, t], dim=1)
                else:
                    x, y = batch

                batch_loss, batch_grad, bs = self._train_one_batch(x, y, first)
                total_samples += bs
                # Debug: verify data ordering consistency (disabled)
                # if self.epoch_num <= 2 and total_samples <= 20000:  # Only first few batches
                #     if dist.is_initialized():
                #         print(f"E{self.epoch_num} P{dist.get_rank()}: first_5_labels={y[:5].tolist()}, batch_size={bs}")
                #     else:
                #         print(f"E{self.epoch_num} P0: first_5_labels={y[:5].tolist()}, batch_size={bs}")
                # print(
                #     f"Process {dist.get_rank()}: batch_size={len(y)}, total_samples_so_far={total_samples}"
                # )

                # weight by global sample count (optimized with pre-computed scaling factor)
                epoch_loss += batch_loss * bs * self._loss_scaling_factor
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
            if (
                self._lssr1_tr_present() or self._sgd_present()
            ) and self.epoch_num % 3 == 0:
                full = self._eval_full_objective()
                self._adjust_batch_size(full)

            self.compute_accuracy()
            self.epoch_dt = time.time() - self.epoch_time
            self.running_time = time.time() - self.total_start_time
            self.epoch_time = time.time()
            self.trigger_callbacks("on_epoch_end")
            self.epoch_num += 1

    def run_by_epoch_PINN(self):
        """Simplified epoch loop for PINN datasets, with adaptive and callback logic."""
        # ensure model and loss operate in double precision for PINN training
        self.model = self.model.double()
        if hasattr(self.optimizer, "loc_model"):
            # APTS-based optimizers keep a separate local model that
            # is cloned during initialisation.  When switching the main
            # model to ``float64`` we must also convert this local copy,
            # otherwise forward passes inside the optimizer will mix
            # ``float32`` weights with ``float64`` inputs, triggering
            # dtype mismatch errors.
            self.optimizer.loc_model = self.optimizer.loc_model.double()

        if hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(torch.float64)

        # determine how many samples each process should see
        if not self._apts_ip_present():
            if isinstance(self.train_loader.sampler, DistributedSampler):
                # DistributedSampler already shards the dataset
                self.num_training_samples_per_process = (
                    len(self.train_dataset) / self.world_size
                )
            else:
                # full dataset per process (e.g. domain decomposition overlap)
                self.num_training_samples_per_process = len(self.train_dataset)
        else:
            # APTS_IP processes full dataset per process
            self.num_training_samples_per_process = len(self.train_dataset)

        self.total_start_time = time.time()
        self.epoch_time = time.time()
        self.epoch_num = 0

        while self.epoch_num <= self.config.epochs:
            # per-epoch bookkeeping
            epoch_loss = 0.0
            total_samples = 0
            it = iter(self.train_loader)
            first = self.epoch_num == 0

            # reproducible shuffling for DistributedSampler
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch_num)

            # loop until this rank has seen its shard (plus any overlap)
            while total_samples < self.num_training_samples_per_process:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    batch = next(it)

                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, t, y = batch
                    x = torch.cat([x, t], dim=1)
                else:
                    x, y = batch

                # training step (with first-batch warm-up)
                batch_loss, batch_grad, bs = self._train_one_batch_PINN(x, y, first)
                total_samples += bs

                # weight loss by global or local sample count (optimized with pre-computed scaling factor)
                epoch_loss += batch_loss * bs * self._loss_scaling_factor
                self.loss = epoch_loss

                # handle optimizer-driven batch repeats
                stay = self._stay_here()
                if stay:
                    it = chain([(x, y)], it)

                # adaptive batch-size adjustment for ASNTR/APTS
                if self._asntr_present():
                    self._adjust_batch_size(self.loss)

            # optional full-objective adjustment every 3 epochs
            if (
                self._lssr1_tr_present() or self._sgd_present()
            ) and self.epoch_num % 3 == 0:
                full = self._eval_full_objective()
                self._adjust_batch_size(full)

            # PINNs do not compute accuracy
            self.accuracy = float("nan")

            # timing and callbacks
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
