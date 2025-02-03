import torch
from torchvision import datasets, transforms

from src.datasets.base_dataset import *


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
        C.input_channels = 3
        C.output_classes = 10
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        
        if self.transform is None:
            # Define transformations
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        
        if self.data is None:
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

   