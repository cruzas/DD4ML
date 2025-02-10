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
        C.filename = 'tinyshakespeare.txt'
        C.vocab_size = None
        C.block_size = 128
        C.download = False  # Set to True if implementing download logic
        return C

    def __init__(self, config, data=None):
        super().__init__(config, data)
        
        if data is None:
            # Get the directory of this file and append the relative path to the data file
            file_path = os.path.join(os.path.dirname(__file__), self.config.root, self.config.filename)
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
        self.block_size = self.config.block_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
