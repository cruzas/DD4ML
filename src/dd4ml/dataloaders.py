from typing import Optional

import os
local_rank = int(os.environ.get("LOCAL_RANK", 0))
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class MockDataset(Dataset):
    # NOTE: First=True means that the real input data and mock output data will be provided
    # First=False means that the mock input data and real output data will be provided
    # First=None means that the mock input and output data will be provided
    def __init__(self, dataset, amount_of_batches=None, device=None, first=True):
        """
        Initializes a MockDataset object.

        Args:
            dataset: The dataset to be used.
            amount_of_batches (optional): The number of batches to be used. Defaults to None.
            device (optional): The device to be used. Defaults to None.
            first (optional): A boolean indicating if it is the first dataset. Defaults to True.
        """
        super(MockDataset, self).__init__()
        self.amount_of_batches = amount_of_batches
        self.dataset = dataset
        self.first = first
        self.device = device

    def __len__(self):
        """
        Returns the number of batches in the dataloader.

        :return: The number of batches.
        :rtype: int
        """
        return self.amount_of_batches

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the item at the specified index.
        """
        if self.first == True:
            return (self.dataset[idx][0], 1)
        elif self.first == False:
            return (1, self.dataset[idx][1])
        else:
            return (1, 1)
        
class GeneralizedDistributedDataLoader(DataLoader):
    def __init__(self, model_handler, dataset, batch_size, shuffle, device='cpu' if not torch.cuda.is_available() else 'cuda', num_workers=0, pin_memory=False, seed=0, **kwargs):
        """
        Initializes the GeneralizedDistributedDataLoader object.

        Args:
            model_handler (list of lists of ...): This variable is generated by the ParallelizedModel class.
            dataset: The dataset to be loaded.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.
            device (str): The device to use. Defaults to 'cpu' if torch.cuda.is_available() is False, otherwise 'cuda'.
            num_workers (int): The number of worker processes. Defaults to 0.
            pin_memory (bool): Whether to pin memory. Defaults to False.
            seed (int): The random seed. Defaults to 0.
            **kwargs: Additional keyword arguments.
            
        E.g.: Supppose len_stage_list = 3 and num_replicas = 2.Then:
        model 0 will be distributed across ranks [0,1,2] with first layer in rank 0 and second layer in rank 1 and so on.
        model 1 will be distributed across ranks [3,4,5] with first layer in rank 3 and second layer in rank 4 and so on.
        """
        if 'drop_last' in kwargs:
            #print a warning 
            print(f"(WARNING) drop_last will always be set to 'True' in the GeneralizedDistributedDataLoader.")
            kwargs.pop('drop_last')
        if batch_size > len(dataset):
            print(f"(WARNING) Batch size {batch_size} is greater than the dataset size {len(dataset)}. Setting batch size to dataset size.")
            batch_size = min(batch_size, len(dataset))
        
        tot_replicas = model_handler.tot_replicas
        num_stages = model_handler.num_stages
        
        first_layer_ranks = model_handler.get_stage_ranks(stage_name='first', mode='global')
        last_layer_ranks = model_handler.get_stage_ranks(stage_name='last', mode='global')
                        
        rank = dist.get_rank()
        if num_stages == 1:
            self.sampler = GeneralizedDistributedSampler(layer_ranks=first_layer_ranks, dataset=dataset, num_replicas=tot_replicas, rank=rank, shuffle=shuffle, drop_last=True, seed=seed, **kwargs)
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size//tot_replicas, shuffle=False, sampler=self.sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, **kwargs)
        elif rank not in first_layer_ranks+last_layer_ranks: # rank in the middle does not require any real data
            # Make a mock dataset with the same amount of batches as the original dataset (this is needed to keep iterations consistent across all ranks)
            amount_of_batches = 1 if len(dataset) == batch_size else len(dataset) // batch_size
            dataset = MockDataset(dataset, amount_of_batches, device=device, first=None)     
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, **kwargs)    
        elif rank in first_layer_ranks:
            dataset = MockDataset(dataset, len(dataset), device=device, first=True)
            self.sampler = GeneralizedDistributedSampler(layer_ranks=first_layer_ranks, dataset=dataset, num_replicas=tot_replicas, rank=rank, shuffle=shuffle, drop_last=True, seed=seed, **kwargs)
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size//tot_replicas, shuffle=False, sampler=self.sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, **kwargs)
        else:
            dataset = MockDataset(dataset, len(dataset), device=device, first=False)
            self.sampler = GeneralizedDistributedSampler(layer_ranks=last_layer_ranks, dataset=dataset, num_replicas=tot_replicas, rank=rank, shuffle=shuffle, drop_last=True, seed=seed, **kwargs)
            super(GeneralizedDistributedDataLoader, self).__init__(dataset=dataset, batch_size=batch_size//tot_replicas, shuffle=False, sampler=self.sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=True, **kwargs)

class GeneralizedDistributedSampler(DistributedSampler):
    def __init__(self, layer_ranks, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, **kwargs): 
        """
        Initializes the GeneralizedDistributedSampler object.

        Parameters:
        - layer_ranks: List of ranks for each layer.
        - dataset: The dataset to sample from.
        - num_replicas: Number of distributed replicas. Defaults to None.
        - rank: Rank of the current process. Defaults to None.
        - shuffle: Whether to shuffle the samples. Defaults to True.
        - seed: Seed value for shuffling. Defaults to 0.
        - drop_last: Whether to drop the last incomplete batch. Defaults to False.
        - **kwargs: Additional keyword arguments.

        Raises:
        - RuntimeError: If the distributed package is not available.
        - ValueError: If num_replicas is not equal to the number of layer_ranks.
        """
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank() if rank is None else rank
        if num_replicas is not None and (len(layer_ranks) != num_replicas):
            raise ValueError("num_replicas should be equal to the number of first_layer_ranks.")
        rank = layer_ranks.index(rank)
        kwargs.update({'dataset': dataset, 'num_replicas': len(layer_ranks), 'rank': rank, 'shuffle': shuffle, 'seed': seed, 'drop_last': drop_last})
        super(GeneralizedDistributedSampler, self).__init__(**kwargs)
        # super(GeneralizedDistributedSampler, self).__init__(dataset=dataset, num_replicas=len(first_layer_ranks), rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last, **kwargs)


import time
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def change_channel_position(tensor):
    # TODO: change this so that it's adaptive to "modified" datasets
    flattened_tensor = tensor
    if len(tensor.shape)==3: #image Black and white
        flattened_tensor = tensor.unsqueeze(1) # Adds a dimension to the tensor
    elif 3 in tensor.shape[1:] and len(tensor.shape[1:])==3: #image RGB
        flattened_tensor = tensor.permute(0, 3, 1, 2) # Changes the position of the channels
    elif 4 in tensor.shape[1:]: #TODO: video?
        raise ValueError('TODO')
    # Now the shape is correct, check with ---> plt.imshow(flattened_tensor[0,1,:,:].cpu())
    return flattened_tensor

def normalize_dataset(data, mean=[], std=[]):
    if not mean:
        mean = torch.mean(data, dtype=torch.float32) # Calculate the mean and standard deviation of the dataset
    if not std:
        std = torch.std(data)
    data_normalized = (data - mean) / std # Normalize the dataset
    return data_normalized


class Power_DL():
    def __init__(self, 
                 dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 device=torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu'), 
                 precision=torch.get_default_dtype(), 
                 overlapping_samples=0, 
                 SHARED_OVERLAP=False, # if True: overlap samples are shared between minibatches
                 mean=[], 
                 std=[]):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.iter = 0
        self.epoch = 0
        self.precision = int(''.join([c for c in str(precision) if c.isdigit()]))
        self.overlap = overlapping_samples
        self.SHARED_OVERLAP = SHARED_OVERLAP
        self.minibatch_amount = int(np.ceil(len(self.dataset)/self.batch_size))
        if self.minibatch_amount == 1:
            if self.overlap !=0:
                print('(Power_DL) Warning: overlap is not used. Only 1 minibatch (full dataset).')
            self.overlap = 0

        if "Subset" in str(dataset.__class__):
            self.dataset = dataset.dataset

        if 'numpy' in str(self.dataset.data.__class__):
            self.dataset.data = torch.from_numpy(self.dataset.data)

        if round(self.overlap) != self.overlap and self.overlap<1 and self.overlap>0: #overlap is a percentage
            self.overlap = int(self.overlap*self.batch_size)
        if self.overlap == self.batch_size:
            raise ValueError('Overlap cannot be equal to the minibatch size, this will generate "mini"batches with the entire dataframe each.')
        elif self.overlap > self.batch_size:
            raise ValueError('Overlap cannot be higher than minibatch size.')
        
        assert 'torch' in str(self.dataset.data.__class__)
        self.dataset.data = self.dataset.data.to(self.device)
        number = int(''.join([c for c in str(self.dataset.data.dtype) if c.isdigit()]))
        if self.precision != number:
            exec(f"self.dataset.data = self.dataset.data.to(torch.float{self.precision})")

        self.dataset.data = change_channel_position(self.dataset.data)
        # self.dataset.data = normalize_dataset(self.dataset.data, mean, std) # Data normalization

        dtype = torch.LongTensor
        if torch.cuda.is_available() and ('MNIST' in str(dataset.__class__) or 'CIFAR' in str(dataset.__class__)):
            dtype = torch.cuda.LongTensor
        elif torch.cuda.is_available() and 'Sine' in str(dataset.__class__):
            dtype = torch.cuda.FloatTensor
        elif not torch.cuda.is_available() and ('MNIST' in str(dataset.__class__) or 'CIFAR' in str(dataset.__class__)):
            dtype = torch.LongTensor
        elif not torch.cuda.is_available() and 'Sine' in str(dataset.__class__):
            if self.precision == 32:
                dtype = torch.float32
            elif self.precision == 64:
                dtype = torch.float64
            elif self.precision == 16:
                dtype = torch.float16

        try:
            # TODO: Copy on cuda to avoid problems with parallelization (and maybe other problems)
            # self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets.cpu())).type(torch.LongTensor).to(self.device) 
            self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets.cpu())).to(self.device).type(dtype)
        except:
            # self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).type(torch.LongTensor).to(self.device)
            # if torch.cuda.is_available():
            #     self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).to(self.device).type(dtype)
            # else:
            self.dataset.targets = torch.from_numpy(np.array(self.dataset.targets)).to(self.device).type(dtype)



    def __iter__(self):
        g = torch.Generator(device=self.device)
        g.manual_seed(self.epoch * 100)
        self.indices = torch.randperm(len(self.dataset), generator=g, device=self.device) if self.shuffle else torch.arange(len(self.dataset), device=self.device)
        self.epoch += 1
        self.iter = 0
        return self
    

    def __next__(self):
        index_set = self.indices[ self.iter*self.batch_size : self.iter*self.batch_size+self.batch_size ]
        self.iter += 1
        if len(index_set) == 0:
            raise StopIteration()
        
        # This is probably slow, it would be better to generate the overlapping indices in the __init__ method
        if self.overlap > 0:
            overlapping_indices = torch.tensor([], dtype=torch.long, device=self.device)
            for i in range(self.minibatch_amount):
                if i != self.iter:
                    if self.SHARED_OVERLAP:
                        indexes = torch.tensor([range(i*self.batch_size, i*self.batch_size+self.overlap)], device=self.device)
                    else:
                        indexes = torch.randint(i*self.batch_size, i*self.batch_size+self.batch_size, (self.overlap,), device=self.device) # generate "self.overlap" random indeces inside the i-th minibatch
                    overlapping_indices = torch.cat([overlapping_indices, self.indices[indexes]], 0)
            
            index_set = torch.cat([index_set, overlapping_indices], 0) # Combining the original index set with the overlapping indices

        return self.dataset.data[index_set], self.dataset.targets[index_set]
        