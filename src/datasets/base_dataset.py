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