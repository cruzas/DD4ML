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

from dd4ml.dataloaders import MicroBatchOverlapSampler, OverlapBatchSampler


def test_micro_overlap_verbose():
    # Synthetic dataset of indices 0…9
    base = SequentialSampler(list(range(10)))

    # OverlapBatchSampler: batch_size=4, overlap=2
    obs = OverlapBatchSampler(base, batch_size=4, overlap=0.5, drop_last=True)
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
    obs = OverlapBatchSampler(base, batch_size=4, overlap=2, drop_last=True)
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
    batch_size = 17
    overlap = 0.5

    obs_m = OverlapBatchSampler(
        base_m, batch_size=batch_size, overlap=overlap, drop_last=True
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
