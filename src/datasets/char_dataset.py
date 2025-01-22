import torch
from torch.utils.data.dataloader import DataLoader

from src.datasets.base_dataset import BaseDataset
from src.utils import CfgNode as CN
from src.utils import dprint


class CharDataset(BaseDataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        super().__init__(config, data)

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        dprint(f'data has {data_size} characters, {vocab_size} unique.')

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def get_sample_input(self, config):
        dummy_train_loader = DataLoader(
            self,
            sampler=torch.utils.data.RandomSampler(
                self, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=1,
            num_workers=config.num_workers,
        )

        x_batch, _ = next(iter(dummy_train_loader))
        device = config.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return x_batch.to(device)

    def get_sample_target(self, config):
        dummy_train_loader = DataLoader(
            self,
            sampler=torch.utils.data.RandomSampler(
                self, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=1,
            num_workers=config.num_workers,
        )

        _, y_batch = next(iter(dummy_train_loader))
        device = config.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return y_batch.to(device)
