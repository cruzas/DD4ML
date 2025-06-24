# overlap_sampler.py
import math
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
)
from torch.utils.data.distributed import DistributedSampler


class OverlapBatchSampler(BatchSampler):
    """
    Successive batches share `overlap` examples **without** changing the number of batches,
    and treat mini-batches circularly when drop_last=True (last overlaps first).
    stride = batch_size # unique indices per step
    size = batch_size + overlap_sz # actual minibatch length (except final partial when drop_last=False)
    """

    def __init__(
        self,
        base_sampler: Sampler[int],
        batch_size: int,
        overlap: float | int = 0.0,
        drop_last: bool = False,
    ):
        self.base_sampler = base_sampler
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

        if isinstance(overlap, float) and 0.0 < overlap < 1.0:
            self.overlap_sz = int(round(overlap * self.batch_size))
        elif isinstance(overlap, int) and overlap >= 1:
            self.overlap_sz = min(overlap, self.batch_size - 1)
        else:
            self.overlap_sz = 0

        if self.overlap_sz >= self.batch_size:
            raise ValueError("`overlap` must be smaller than `batch_size`")

    def __iter__(self):
        idxs = list(self.base_sampler)
        n = len(idxs)
        stride = self.batch_size
        n_batches = n // stride if self.drop_last else math.ceil(n / stride)

        for b in range(n_batches):
            start = b * stride
            # unique portion (wrap if needed)
            unique = idxs[start : start + stride]
            if len(unique) < stride:
                unique += idxs[: stride - len(unique)]

            # if non-drop-last final partial batch, omit overlap
            # if not self.drop_last and (start + stride) >= n:
            if not self.drop_last and (start + stride) > n:
                yield unique
                continue

            # compute overlap
            if self.drop_last and b == (n_batches - 1):
                # last batch in drop_last mode: overlap comes from the start of dataset
                overlap = idxs[0 : self.overlap_sz]
            else:
                o_start = (start + stride) % n
                o_end = o_start + self.overlap_sz
                if o_end <= n:
                    overlap = idxs[o_start:o_end]
                else:
                    overlap = idxs[o_start:] + idxs[: o_end - n]

            yield unique + overlap

    def __len__(self):
        n = len(self.base_sampler)
        return (
            n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        )


class MicroBatchOverlapSampler:
    """
    Splits each mini-batch from OverlapBatchSampler into N subdomains,
    preserving overlap within each subdomain. Works with variable last batch size.
    """

    def __init__(
        self,
        overlap_sampler: OverlapBatchSampler,
        num_subdomains: int,
        allow_empty_microbatches: bool = False,
    ):
        if not isinstance(overlap_sampler, OverlapBatchSampler):
            raise TypeError(
                "overlap_sampler must be an instance of OverlapBatchSampler"
            )
        if num_subdomains <= 0:
            raise ValueError("num_subdomains must be positive")
        if overlap_sampler.overlap_sz == 0:
            raise ValueError(
                "OverlapBatchSampler must have overlap > 0 for micro-batch overlap"
            )

        total_size = overlap_sampler.batch_size + overlap_sampler.overlap_sz
        if num_subdomains > total_size and not allow_empty_microbatches:
            raise ValueError(
                f"num_subdomains ({num_subdomains}) > total mini-batch size ({total_size})."
            )

        self.overlap_sampler = overlap_sampler
        self.num_subdomains = num_subdomains
        self.allow_empty_microbatches = allow_empty_microbatches
        self.mini_batches = list(overlap_sampler)
        if len(self.mini_batches) == 0:
            raise ValueError("OverlapBatchSampler produced no mini-batches")

    def __iter__(self) -> Iterator[List[List[int]]]:
        for mini_batch in self.mini_batches:
            yield self._split_mini_batch(mini_batch)

    def _split_mini_batch(self, mini_batch: List[int]) -> List[List[int]]:
        unique_size = min(len(mini_batch), self.overlap_sampler.batch_size)
        unique = mini_batch[:unique_size]
        overlap = mini_batch[unique_size:]
        micro_batches = []
        for j in range(self.num_subdomains):
            mb = [unique[k] for k in range(j, len(unique), self.num_subdomains)]
            mb += [overlap[k] for k in range(j, len(overlap), self.num_subdomains)]
            micro_batches.append(mb)
        if not self.allow_empty_microbatches:
            empties = sum(1 for mb in micro_batches if not mb)
            if empties:
                raise RuntimeError(f"{empties} empty micro-batches generated.")
        return micro_batches

    def __len__(self):
        return len(self.mini_batches)

    def get_overlap_info(self) -> dict:
        info = {
            "mini_batches": len(self.mini_batches),
            "subdomains": self.num_subdomains,
            "overlap_per_minibatch": self.overlap_sampler.overlap_sz,
            "allow_empty_microbatches": self.allow_empty_microbatches,
        }
        if len(self.mini_batches) >= 2:
            m0 = self._split_mini_batch(self.mini_batches[0])
            m1 = self._split_mini_batch(self.mini_batches[1])
            overlaps = [
                len(set(m0[j]) & set(m1[j])) for j in range(self.num_subdomains)
            ]
            info.update(
                {
                    "actual_microbatch_overlaps": overlaps,
                    "first_minibatch_sizes": [len(m) for m in m0],
                    "second_minibatch_sizes": [len(m) for m in m1],
                }
            )
        else:
            info["note"] = "Need ≥2 mini-batches to calc overlap."
        return info


class GeneralizedDistributedSampler(DistributedSampler):
    def __init__(
        self,
        layer_ranks,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Initializes the GeneralizedDistributedSampler object.

        Parameters:
        - layer_ranks: List of ranks for each layer.
        - dataset: The dataset to sample from.
        - num_replicas: Number of distributed replicas. Defaults to None.
        - rank: Rank of the current process. Defaults to None.
        - shuffle: Whether to shuffle the samples. Defaults to True.
        - seed: Seed value for shuffling. Defaults to 0.
        - drop_last: Whether to drop the last incomplete batch. Defaults to False.
        - **kwargs: Additional keyword arguments.

        Raises:
        - RuntimeError: If the distributed package is not available.
        - ValueError: If num_replicas is not equal to the number of layer_ranks.
        """
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank() if rank is None else rank
        if num_replicas is not None and (len(layer_ranks) != num_replicas):
            raise ValueError(
                "num_replicas should be equal to the number of first_layer_ranks."
            )
        rank = layer_ranks.index(rank)
        kwargs.update(
            {
                "dataset": dataset,
                "num_replicas": len(layer_ranks),
                "rank": rank,
                "shuffle": shuffle,
                "seed": seed,
                "drop_last": drop_last,
            }
        )
        super(GeneralizedDistributedSampler, self).__init__(**kwargs)
        # super(GeneralizedDistributedSampler, self).__init__(dataset=dataset, num_replicas=len(first_layer_ranks), rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last, **kwargs)


def change_channel_position(tensor):
    # TODO: change this so that it's adaptive to "modified" datasets
    flattened_tensor = tensor
    if len(tensor.shape) == 3:  # image Black and white
        flattened_tensor = tensor.unsqueeze(1)  # Adds a dimension to the tensor
    elif 3 in tensor.shape[1:] and len(tensor.shape[1:]) == 3:  # image RGB
        # Changes the position of the channels
        flattened_tensor = tensor.permute(0, 3, 1, 2)
    elif 4 in tensor.shape[1:]:  # TODO: video?
        raise ValueError("TODO")
    # Now the shape is correct, check with ---> plt.imshow(flattened_tensor[0,1,:,:].cpu())
    return flattened_tensor


def normalize_dataset(data, mean=[], std=[]):
    if not mean:
        # Calculate the mean and standard deviation of the dataset
        mean = torch.mean(data, dtype=torch.float32)
    if not std:
        std = torch.std(data)
    data_normalized = (data - mean) / std  # Normalize the dataset
    return data_normalized


class Power_DL:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        precision=torch.get_default_dtype(),
        overlapping_samples=0,
        SHARED_OVERLAP=False,  # if True: overlap samples are shared between minibatches
        mean=[],
        std=[],
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.iter = 0
        self.epoch = 0
        self.precision = int("".join([c for c in str(precision) if c.isdigit()]))
        self.overlap = overlapping_samples
        self.SHARED_OVERLAP = SHARED_OVERLAP
        self.minibatch_amount = int(np.ceil(len(self.dataset) / self.batch_size))
        if self.minibatch_amount == 1:
            if self.overlap != 0:
                print(
                    "(Power_DL) Warning: overlap is not used. Only 1 minibatch (full dataset)."
                )
            self.overlap = 0

        if "Subset" in str(dataset.__class__):
            self.dataset = dataset.dataset

        if "numpy" in str(self.dataset.data.__class__):
            self.dataset.data = torch.from_numpy(self.dataset.data)

        # overlap is a percentage
        if (
            round(self.overlap) != self.overlap
            and self.overlap < 1
            and self.overlap > 0
        ):
            self.overlap = int(self.overlap * self.batch_size)
        if self.overlap == self.batch_size:
            raise ValueError(
                'Overlap cannot be equal to the minibatch size, this will generate "mini"batches with the entire dataframe each.'
            )
        elif self.overlap > self.batch_size:
            raise ValueError("Overlap cannot be higher than minibatch size.")

        assert "torch" in str(self.dataset.data.__class__)
        self.dataset.data = self.dataset.data.to(self.device)
        number = int("".join([c for c in str(self.dataset.data.dtype) if c.isdigit()]))
        if self.precision != number:
            exec(
                f"self.dataset.data = self.dataset.data.to(torch.float{self.precision})"
            )

        self.dataset.data = change_channel_position(self.dataset.data)
        # self.dataset.data = normalize_dataset(self.dataset.data, mean, std) # Data normalization

        dtype = torch.LongTensor
        if torch.cuda.is_available() and (
            "MNIST" in str(dataset.__class__) or "CIFAR" in str(dataset.__class__)
        ):
            dtype = torch.cuda.LongTensor
        elif torch.cuda.is_available() and "Sine" in str(dataset.__class__):
            dtype = torch.cuda.FloatTensor
        elif not torch.cuda.is_available() and (
            "MNIST" in str(dataset.__class__) or "CIFAR" in str(dataset.__class__)
        ):
            dtype = torch.LongTensor
        elif not torch.cuda.is_available() and "Sine" in str(dataset.__class__):
            if self.precision == 32:
                dtype = torch.float32
            elif self.precision == 64:
                dtype = torch.float64
            elif self.precision == 16:
                dtype = torch.float16

        try:
            # TODO: Copy on cuda to avoid problems with parallelization (and maybe other problems)
            # self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets.cpu())).type(torch.LongTensor).to(self.device)
            self.dataset.targets = (
                torch.from_numpy(np.array(self.dataset.targets.cpu()))
                .to(self.device)
                .type(dtype)
            )
        except:
            # self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).type(torch.LongTensor).to(self.device)
            # if torch.cuda.is_available():
            #     self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).to(self.device).type(dtype)
            # else:
            self.dataset.targets = (
                torch.from_numpy(np.array(self.dataset.targets))
                .to(self.device)
                .type(dtype)
            )

    def __iter__(self):
        g = torch.Generator(device=self.device)
        g.manual_seed(self.epoch * 100)
        self.indices = (
            torch.randperm(len(self.dataset), generator=g, device=self.device)
            if self.shuffle
            else torch.arange(len(self.dataset), device=self.device)
        )
        self.epoch += 1
        self.iter = 0
        return self

    def __next__(self):
        index_set = self.indices[
            self.iter * self.batch_size : self.iter * self.batch_size + self.batch_size
        ]
        self.iter += 1
        if len(index_set) == 0:
            raise StopIteration()

        # This is probably slow, it would be better to generate the overlapping indices in the __init__ method
        if self.overlap > 0:
            overlapping_indices = torch.tensor([], dtype=torch.long, device=self.device)
            for i in range(self.minibatch_amount):
                if i != self.iter:
                    if self.SHARED_OVERLAP:
                        indexes = torch.tensor(
                            [
                                range(
                                    i * self.batch_size,
                                    i * self.batch_size + self.overlap,
                                )
                            ],
                            device=self.device,
                        )
                    else:
                        # generate "self.overlap" random indeces inside the i-th minibatch
                        indexes = torch.randint(
                            i * self.batch_size,
                            i * self.batch_size + self.batch_size,
                            (self.overlap,),
                            device=self.device,
                        )
                    overlapping_indices = torch.cat(
                        [overlapping_indices, self.indices[indexes]], 0
                    )

            # Combining the original index set with the overlapping indices
            index_set = torch.cat([index_set, overlapping_indices], 0)

        return self.dataset.data[index_set], self.dataset.targets[index_set]
