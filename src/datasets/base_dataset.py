from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from src.utils import *

CN = CfgNode

class BaseDataset(Dataset, ABC):

    @staticmethod
    def get_default_config():
        C = CN()
        C.train = True
        C.download = True
        C.root = "../rawdata/"
        C.percentage = 100.0 # percentage of the dataset to use
        return C

    def __init__(self, config, data, transform):
        self.config = config
        self.data = data
        self.transform

    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    def compute_class_weights(self):
        # Count the occurrences of each class in the dataset
        labels = [label for _, label in self]
        class_counts = Counter(labels)
        total_samples = sum(class_counts.values())

        # Compute class weights as the inverse of the frequency
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

        # Convert to tensor and normalize
        weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float)
        return weights / weights.sum()
    
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