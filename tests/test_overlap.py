import math
from typing import Iterator, List

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
    SequentialSampler,
    TensorDataset,
)
from torchvision import datasets, transforms


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


def test_micro_overlap_verbose():
    # Synthetic dataset of indices 0…9
    base = SequentialSampler(list(range(10)))

    # OverlapBatchSampler: batch_size=4, overlap=2
    obs = OverlapBatchSampler(base, batch_size=4, overlap=0.5, drop_last=False)
    # MicroBatchOverlapSampler with 2 subdomains
    mbs = MicroBatchOverlapSampler(obs, num_subdomains=2)

    mini_batches = list(obs)
    micro_batches = list(mbs)

    # Print mini‑batches and their micro‑batches
    for i, (mini, micros) in enumerate(zip(mini_batches, micro_batches)):
        print(f"Mini-batch {i}: {mini}")
        for j, mb in enumerate(micros):
            print(f"  Micro-batch [subdomain {j}]: {mb}")
        print()

    # Report expected vs actual overlaps
    expected_overlap = obs.overlap_sz // mbs.num_subdomains
    print(f"Expected overlap per subdomain: {expected_overlap}\n")

    for i in range(len(micro_batches) - 1):
        for sd in range(mbs.num_subdomains):
            inter = set(micro_batches[i][sd]) & set(micro_batches[i + 1][sd])
            print(
                f"Actual overlap between mini-batch {i} → {i+1}, "
                f"subdomain {sd}: {len(inter)}"
            )


# ---------------------- Tests ----------------------
if __name__ == "__main__":
    # Simple synthetic test
    base = SequentialSampler(list(range(10)))
    obs = OverlapBatchSampler(base, batch_size=4, overlap=2, drop_last=False)
    mbs = MicroBatchOverlapSampler(obs, num_subdomains=2)

    print("Synthetic batches and micro-batches:")
    for i, mini in enumerate(obs):
        print(f"Mini-batch {i}: {mini}")
        micros = mbs._split_mini_batch(mini)
        print(f"  Micro-batches: {micros}")
    print()

    # MNIST test with specific overlap and micro-batch info
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    base_m = SequentialSampler(mnist)
    batch_size = 16
    overlap = 0.5

    obs_m = OverlapBatchSampler(
        base_m, batch_size=batch_size, overlap=overlap, drop_last=False
    )
    mbs_m = MicroBatchOverlapSampler(obs_m, num_subdomains=2)
    loader = DataLoader(mnist, batch_sampler=obs_m)

    print(f"MNIST test with overlap={overlap} (overlap_sz={obs_m.overlap_sz}):")
    # Print first two batch sizes
    batch0_imgs, _ = next(iter(loader))
    print(f"Batch 0 size={batch0_imgs.shape[0]}")
    # advance and print second
    batch1_imgs, _ = next(iter(loader))
    print(f"Batch 1 size={batch1_imgs.shape[0]}")

    # Print micro-batch details for first two mini-batches
    mini0 = mbs_m.mini_batches[0]
    micros0 = mbs_m._split_mini_batch(mini0)
    print(f"Mini-batch 0 indices: {mini0}")
    print(f"  Micro-batches: {micros0}")

    mini1 = mbs_m.mini_batches[1]
    micros1 = mbs_m._split_mini_batch(mini1)
    print(f"Mini-batch 1 indices: {mini1}")
    print(f"  Micro-batches: {micros1}")

    # Print last mini-batch and its micro-batches
    last_idx = len(mbs_m.mini_batches) - 1
    mini_last = mbs_m.mini_batches[last_idx]
    micros_last = mbs_m._split_mini_batch(mini_last)
    print(f"Last mini-batch {last_idx} indices: {mini_last}")
    print(f"  Micro-batches: {micros_last}")

    print("MNIST micro-batch test passed.")
