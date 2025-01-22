from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from src.utils import CfgNode as CN


class BaseDataset(Dataset, ABC):

    @staticmethod
    def get_default_config():
        C = CN()
        C.percentage = 100.0
        return C

    def __init__(self, config, data):
        self.config = config
        self.data = data

    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass