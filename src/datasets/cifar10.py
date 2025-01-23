import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.datasets.base_dataset import BaseDataset
from src.utils import CfgNode as CN
from src.utils import dprint


class CIFAR10Dataset(BaseDataset):
    """
    CIFAR-10 dataset class following BaseDataset structure
    """

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.root = '../rawdata/'
        C.train = True
        C.download = True
        return C

    def __init__(self, config, data=None):
        super().__init__(config, data)
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        # Load CIFAR-10 dataset
        self.data = datasets.CIFAR10(
            root=self.config.root, 
            train=self.config.train, 
            download=self.config.download, 
            transform=self.transform
        )
        
        self.classes = self.data.classes
        
        dprint(f'CIFAR-10 dataset loaded with {len(self.data)} images, {len(self.classes)} classes.')

    def get_input_channels(self):
        return 3  # RGB images

    def get_output_classes(self):
        return len(self.classes)

    def get_block_size(self):
        return 32  # Image size for CIFAR-10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

    def get_sample_input(self, config):
        dummy_train_loader = DataLoader(
            self,
            sampler=torch.utils.data.RandomSampler(
                self, replacement=True, num_samples=int(1e10)
            ),
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
                self, replacement=True, num_samples=int(1e10)
            ),
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