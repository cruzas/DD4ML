import os

import torch

from src.datasets.base_dataset import *


class TinyShakespeareDataset(BaseDataset):
    """
    TinyShakespeare dataset class following BaseDataset structure.
    """

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.root = '../rawdata/'
        C.filename = 'tinyshakespeare.txt'
        C.block_size = 128
        C.download = False  # Set to True if implementing download logic
        return C

    def __init__(self, config, data=None):
        super().__init__(config, data)
        
        if data is None:
            file_path = os.path.join(self.config.root, self.config.filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = f.read()
        else:
            self.data = data
        
        chars = sorted(list(set(self.data)))
        data_size, vocab_size = len(self.data), len(chars)
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
        chunk = self.data[idx: idx + self.config.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
