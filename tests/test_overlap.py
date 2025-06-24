import math
from typing import Iterator, List, Union

from torch.utils.data import BatchSampler, Sampler
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
        overlap: Union[float, int] = 0.0,
        drop_last: bool = False,
    ):
        # Don't call super().__init__ to avoid conflicts
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

        # Cache the indices to ensure consistency across multiple iterations
        self._cached_indices = None

    def _get_indices(self):
        """Get cached indices or generate them if not cached."""
        if self._cached_indices is None:
            self._cached_indices = list(self.base_sampler)
        return self._cached_indices

    def __iter__(self) -> Iterator[List[int]]:
        idxs = self._get_indices()
        n = len(idxs)

        if n == 0:
            return

        stride = self.batch_size
        n_batches = n // stride if self.drop_last else math.ceil(n / stride)

        for b in range(n_batches):
            start = b * stride

            # Get unique portion
            end = min(start + stride, n)
            unique = idxs[start:end]

            # If we need to pad the unique portion (only in drop_last mode)
            if self.drop_last and len(unique) < stride:
                needed = stride - len(unique)
                unique += idxs[:needed]

            # Handle final partial batch without overlap
            if not self.drop_last and end >= n and len(unique) < stride:
                yield unique
                continue

            # Compute overlap
            if self.overlap_sz > 0:
                if self.drop_last and b == (n_batches - 1):
                    # Last batch in drop_last mode: overlap comes from the start
                    overlap = idxs[: self.overlap_sz]
                else:
                    # Normal overlap from next positions
                    o_start = end % n
                    overlap = []
                    for i in range(self.overlap_sz):
                        overlap.append(idxs[(o_start + i) % n])

                yield unique + overlap
            else:
                yield unique

    def __len__(self):
        if self._cached_indices is None:
            n = len(self.base_sampler)
        else:
            n = len(self._cached_indices)
        return (
            n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        )


class MicroBatchOverlapSampler:
    """
    Splits each mini-batch from OverlapBatchSampler into N subdomains,
    preserving overlap within each subdomain. Works with variable last batch size.

    Note: This is NOT a BatchSampler - it's a wrapper that yields lists of micro-batches.
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

    def __iter__(self) -> Iterator[List[List[int]]]:
        for mini_batch in self.overlap_sampler:
            yield self._split_mini_batch(mini_batch)

    def _split_mini_batch(self, mini_batch: List[int]) -> List[List[int]]:
        unique_size = min(len(mini_batch), self.overlap_sampler.batch_size)
        unique = mini_batch[:unique_size]
        overlap = mini_batch[unique_size:]

        micro_batches = []
        for j in range(self.num_subdomains):
            # Distribute unique indices round-robin
            mb_unique = [unique[k] for k in range(j, len(unique), self.num_subdomains)]
            # Distribute overlap indices round-robin
            mb_overlap = [
                overlap[k] for k in range(j, len(overlap), self.num_subdomains)
            ]
            mb = mb_unique + mb_overlap
            micro_batches.append(mb)

        if not self.allow_empty_microbatches:
            empties = sum(1 for mb in micro_batches if not mb)
            if empties:
                raise RuntimeError(f"{empties} empty micro-batches generated.")

        return micro_batches

    def __len__(self):
        return len(self.overlap_sampler)

    def get_overlap_info(self) -> dict:
        """Get information about overlap configuration and actual overlaps."""
        info = {
            "mini_batches": len(self.overlap_sampler),
            "subdomains": self.num_subdomains,
            "overlap_per_minibatch": self.overlap_sampler.overlap_sz,
            "allow_empty_microbatches": self.allow_empty_microbatches,
        }

        # Get first two mini-batches to analyze overlap
        mini_batches = list(self.overlap_sampler)
        if len(mini_batches) >= 2:
            m0 = self._split_mini_batch(mini_batches[0])
            m1 = self._split_mini_batch(mini_batches[1])
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


class MicroBatchFlattenSampler(BatchSampler):
    """
    A BatchSampler that flattens micro-batches from MicroBatchOverlapSampler
    into individual batches for use with DataLoader.
    """

    def __init__(self, micro_batch_sampler: MicroBatchOverlapSampler):
        self.micro_batch_sampler = micro_batch_sampler
        # Don't call super().__init__()

    def __iter__(self) -> Iterator[List[int]]:
        for micro_batches in self.micro_batch_sampler:
            for micro_batch in micro_batches:
                if micro_batch:  # Only yield non-empty micro-batches
                    yield micro_batch

    def __len__(self):
        # This is approximate since we don't know how many non-empty micro-batches there will be
        return len(self.micro_batch_sampler) * self.micro_batch_sampler.num_subdomains


# Fixed setup_data_loaders method
def setup_data_loaders(self):
    """(Re)create train and test loaders using current_batch_size"""
    cfg, ds_train, ds_test = self.config, self.train_dataset, self.test_dataset
    bs = self.current_batch_size
    overlap = cfg.overlap if hasattr(cfg, "overlap") else 0

    # Check if batch size equals or exceeds dataset
    if bs >= len(ds_train):
        print(
            f"Batch size {bs} >= dataset size {len(ds_train)}; using full dataset as single batch, overlap=0."
        )
        self.current_batch_size = len(ds_train)
        cfg.overlap = 0
        bs = self.current_batch_size
        overlap = 0

    # Determine distributed context
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if cfg.use_pmw:
        # PMW loader (assuming this is a custom implementation)
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
        shard_size = math.ceil(len(ds_train) / world_size)
        if pp_bs >= shard_size:
            print(
                f"Per-process batch size {pp_bs} >= shard size {shard_size}; shard as single batch, overlap=0."
            )
            overlap = 0

        if pp_bs < 1:
            raise ValueError(
                f"Per-process batch size {pp_bs} < 1; increase global batch size."
            )

        # Base distributed samplers
        base_train = DistributedSampler(
            ds_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        base_test = DistributedSampler(
            ds_test,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

        # Overlap samplers
        train_ov = OverlapBatchSampler(
            base_sampler=base_train,
            batch_size=pp_bs,
            overlap=overlap,
            drop_last=False,
        )
        test_ov = OverlapBatchSampler(
            base_sampler=base_test,
            batch_size=pp_bs,
            overlap=overlap,
            drop_last=False,
        )

        # Micro-batch splitting based on cfg.num_subdomains
        num_sub = getattr(cfg, "num_subdomains", 1)

        if num_sub > 1:
            if overlap > 0:
                # Create micro-batch samplers with overlap
                train_micro = MicroBatchOverlapSampler(
                    overlap_sampler=train_ov,
                    num_subdomains=num_sub,
                    allow_empty_microbatches=False,
                )
                test_micro = MicroBatchOverlapSampler(
                    overlap_sampler=test_ov,
                    num_subdomains=num_sub,
                    allow_empty_microbatches=False,
                )

                # Use flatten samplers for DataLoader compatibility
                train_sampler = MicroBatchFlattenSampler(train_micro)
                test_sampler = MicroBatchFlattenSampler(test_micro)
                print(
                    f"Using micro-batching with {num_sub} subdomains and overlap={overlap}"
                )
            else:
                # No overlap but subdomains requested - use regular batch samplers
                print(
                    f"Warning: num_subdomains={num_sub} requested but overlap=0. Using regular batching."
                )
                train_sampler = train_ov
                test_sampler = test_ov
        else:
            # num_subdomains = 1: Use overlap samplers directly (overlap between mini-batches)
            train_sampler = train_ov
            test_sampler = test_ov
            if overlap > 0:
                print(
                    f"Using overlap={overlap} between mini-batches (num_subdomains=1)"
                )
            else:
                print("Using regular batching (no overlap, num_subdomains=1)")

        # DataLoaders
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

        print(f"Number of batches in train_loader: {len(self.train_loader)}")

        # Print debug info
        if num_sub > 1 and overlap > 0:
            print(f"Micro-batching active: {num_sub} subdomains with overlap={overlap}")
            if hasattr(train_sampler, "micro_batch_sampler"):
                overlap_info = train_sampler.micro_batch_sampler.get_overlap_info()
                print(f"Overlap info: {overlap_info}")
        elif num_sub == 1 and overlap > 0:
            print(
                f"Mini-batch overlap active: overlap={overlap} between consecutive mini-batches"
            )
        else:
            print("Regular batching (no overlap or micro-batching)")


# Example usage and testing
# if __name__ == "__main__":
#     import torch
#     from torch.utils.data import DataLoader, TensorDataset

#     # Create a simple dataset
#     data = torch.arange(20)
#     dataset = TensorDataset(data)

#     print("Testing OverlapBatchSampler:")
#     base_sampler = torch.utils.data.SequentialSampler(dataset)
#     overlap_sampler = OverlapBatchSampler(
#         base_sampler=base_sampler, batch_size=5, overlap=0.5, drop_last=False
#     )

#     # Test with DataLoader
#     loader = DataLoader(dataset, batch_sampler=overlap_sampler)

#     print(f"Number of batches: {len(loader)}")
#     for i, batch in enumerate(loader):
#         print(f"Batch {i}: indices {batch[0].tolist()}")

#     print("\nTesting MicroBatchOverlapSampler:")
#     micro_sampler = MicroBatchOverlapSampler(
#         overlap_sampler=overlap_sampler,
#         num_subdomains=2,
#         allow_empty_microbatches=False,
#     )

#     print("Micro-batch structure:")
#     for i, micro_batches in enumerate(micro_sampler):
#         print(f"Mini-batch {i}: {micro_batches}")

#     print("\nOverlap info:")
#     print(micro_sampler.get_overlap_info())

#     print("\nTesting with MicroBatchFlattenSampler:")
#     flatten_sampler = MicroBatchFlattenSampler(micro_sampler)
#     micro_loader = DataLoader(dataset, batch_sampler=flatten_sampler)

#     print(f"Number of micro-batches: {len(micro_loader)}")
#     for i, batch in enumerate(micro_loader):
#         print(f"Micro-batch {i}: indices {batch[0].tolist()}")

#     print("\n" + "=" * 60)
#     print("Testing num_subdomains=1 case (overlap between mini-batches):")
#     print("=" * 60)

#     # Test with num_subdomains = 1 (should just use OverlapBatchSampler directly)
#     overlap_sampler_single = OverlapBatchSampler(
#         base_sampler=torch.utils.data.SequentialSampler(dataset),
#         batch_size=4,
#         overlap=2,
#         drop_last=False,
#     )

#     loader_single = DataLoader(dataset, batch_sampler=overlap_sampler_single)
#     print(f"Number of batches (num_subdomains=1): {len(loader_single)}")

#     batches = []
#     for i, batch in enumerate(loader_single):
#         indices = batch[0].tolist()
#         batches.append(indices)
#         print(f"Mini-batch {i}: indices {indices}")

#     # Show overlap between consecutive mini-batches
#     print("\nOverlap analysis between consecutive mini-batches:")
#     for i in range(len(batches) - 1):
#         overlap_indices = set(batches[i]) & set(batches[i + 1])
#         print(
#             f"Overlap between batch {i} and {i+1}: {sorted(overlap_indices)} ({len(overlap_indices)} items)"
#         )

#     print("\n" + "=" * 60)
#     print("Comparison: num_subdomains=2 vs num_subdomains=1")
#     print("=" * 60)

#     print("num_subdomains=2: Creates micro-batches within each mini-batch")
#     print("num_subdomains=1: Creates overlap between consecutive mini-batches")
#     print("Both cases use the same OverlapBatchSampler as the foundation")

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, SequentialSampler
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    # --- wrap MNIST to return indices ---
    class IndexedMNIST(MNIST):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            return idx, img, label

    # Download and prepare MNIST
    mnist_train = IndexedMNIST(
        root=".", train=True, download=True, transform=ToTensor()
    )

    # Overlap + micro‑batch samplers
    base_sampler = SequentialSampler(mnist_train)
    ov_sampler = OverlapBatchSampler(
        base_sampler, batch_size=16, overlap=0.5, drop_last=False
    )
    micro_sampler = MicroBatchOverlapSampler(
        ov_sampler, num_subdomains=2, allow_empty_microbatches=False
    )
    flat_sampler = MicroBatchFlattenSampler(micro_sampler)

    # DataLoader that yields (idxs, images, labels)
    loader = DataLoader(
        mnist_train, batch_sampler=flat_sampler, num_workers=0, pin_memory=True
    )

    # Materialize indices per mini‑batch and per micro‑batch
    all_minibatches = list(ov_sampler)
    all_microbatches = list(micro_sampler)
    total = len(all_minibatches)
    to_inspect = [0, 1, total - 1]

    # Mapping from mini‑batch → list of DataLoader batches (micro‑batches)
    dl_iter = iter(loader)
    dl_micro_idx = 0
    mb_map = {i: [] for i in to_inspect}

    for mb_idx in range(total):
        for sub in range(micro_sampler.num_subdomains):
            idxs, imgs, labels = next(dl_iter)
            if mb_idx in to_inspect:
                mb_map[mb_idx].append(idxs.tolist())
            dl_micro_idx += 1

    # Print
    for mb_idx in to_inspect:
        print(f"\nMini‑batch {mb_idx} indices ({len(all_minibatches[mb_idx])} items):")
        print(all_minibatches[mb_idx])
        for sub, micro_idxs in enumerate(mb_map[mb_idx]):
            print(f"  Micro‑batch {sub} ({len(micro_idxs)} items): {micro_idxs}")
