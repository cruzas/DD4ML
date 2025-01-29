import torch
from torchvision import datasets, transforms

from src.datasets.base_dataset import *


class MNISTDataset(BaseDataset):
    """
    MNIST dataset class following BaseDataset structure
    """

    @staticmethod
    def get_default_config():
        C = BaseDataset.get_default_config()
        C.root = '../rawdata/'
        C.train = True
        C.download = True
        C.input_channels = 1
        C.output_classes = 10
        return C

    def __init__(self, config, data=None, transform=None):
        super().__init__(config, data, transform)
        
        if data is None:
            # Load MNIST dataset
            self.data = datasets.MNIST(
                root=self.config.root, 
                train=self.config.train, 
                download=self.config.download, 
                transform=self.transform
            )
        
        if transform is None:
            # Define transformations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        self.classes = self.data.classes
        dprint(f'MNIST dataset loaded with {len(self.data)} images, {len(self.classes)} classes.')

    def get_input_channels(self):
        return 1  # black-and-white images

    def get_output_classes(self):
        return len(self.classes)

    def get_block_size(self):
        return 28  # Image size for MNIST

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label
