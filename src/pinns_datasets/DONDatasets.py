import numpy as np
import copy
import sys
import torch
import pyDOE
import random
from torch.utils.data import Dataset
from abc import  abstractmethod
import skopt
from sklearn.decomposition import PCA
import pickle
import wget
import requests
import urllib.request
import torch.distributed as dist

from trainers.Config import *


class DONDataset(Dataset):
    def __init__(self, data_set):
        self.data_set = data_set

    def __len__(self):
        return self.data_set[1].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        branch_in = self.data_set[0][idx, :]
        sol = self.data_set[1][idx, :]

        sample = (torch.tensor(branch_in, dtype=torch.get_default_dtype()), torch.tensor(sol, dtype=torch.get_default_dtype()))

        return sample



class DONDatasetBase(object):
    def __init__(self, train_data_path, test_data_path, eval_PCA = False):

        self.eval_PCA =  eval_PCA

        train_set, train_coordinate    = self.get_data(train_data_path)
        test_set, test_coordinate      = self.get_data(test_data_path)        

        self.train_coordinate = torch.tensor(train_coordinate, dtype=torch.get_default_dtype())
        self.test_coordinate = torch.tensor(test_coordinate, dtype=torch.get_default_dtype())

        if(self.eval_PCA == True):
            pca = PCA(n_components=0.9999).fit(train_set[1])

            self.pca_basis      = torch.tensor(pca.components_.T, dtype=torch.get_default_dtype())
            self.pca_mean       = torch.tensor(pca.mean_, dtype=torch.get_default_dtype())
            
            self.num_pca_basis  = self.pca_basis.shape[1]


        self.train_set   = DONDataset(train_set)
        self.test_set    = DONDataset(test_set)

        sampler_train = torch.utils.data.sampler.BatchSampler(
                        torch.utils.data.sampler.RandomSampler(self.train_set),
                        batch_size=int(len(self.train_set)),
                        drop_last=False)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, sampler=sampler_train)

        # test sampler
        sampler_test =  torch.utils.data.sampler.BatchSampler(
                        torch.utils.data.sampler.RandomSampler(self.test_set),
                        batch_size=int(len(self.test_set)),
                        drop_last=False)


        self.test_loader = torch.utils.data.DataLoader(self.test_set, sampler=sampler_test)


    @abstractmethod
    def get_data(self, filename=None):
        raise NotImplementedError



class AdvectionDONDataset(DONDatasetBase):
    def __init__(self, train_data_path, test_data_path, eval_PCA = False):
        super().__init__(train_data_path, test_data_path, eval_PCA)


    @abstractmethod
    def get_data(self, filename=None):
        nx = 40
        nt = 40
        data = np.load(filename)
        x = data["x"].astype(np.float32)
        t = data["t"].astype(np.float32)
        u = data["u"].astype(np.float32)  # N x nt x nx

        u0 = u[:, 0, :]  # N x nx
        xt = np.vstack((np.ravel(x), np.ravel(t))).T
        # u = u.reshape(-1, nt * nx)
        # return (u0, xt), u
        
        
        u = u.reshape(-1, nt * nx)
        return (u0, u), xt

class AniDiffDONDataset(object):
    def __init__(self, data_path=None, eval_PCA=False, normalize=True):
        self.eval_PCA =  eval_PCA

        def bar_progress(current, total, width=80):
          progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
          sys.stdout.write("\r" + progress_message)
          sys.stdout.flush()


        if((data_path==None and os.path.isfile("Anisotropic_5000_1_40_1.pkl")==False) or (data_path is not None and os.path.isfile(data_path)==False)):
            if(dist.get_rank()==0):
                print("AniDiffDONDataset:: Downloading dataset... ")
                url = 'https://zenodo.org/records/10909052/files/Anisotropic_5000_1_40_1.pkl?download=1'
                
                if(data_path is not None):
                    filename = wget.download(url, data_path, bar=bar_progress)             
                else:
                    filename = wget.download(url, bar=bar_progress)       
        else:      
            print("AniDiffDONDataset:: Dataset found ")


        if(data_path==None):
            data_path="Anisotropic_5000_1_40_1.pkl"


        data = pickle.load(open(data_path, 'rb'))
        train_norms, test_norms = data['feature_norms']
        self.train_coordinate = torch.tensor(data['nodes'], dtype=torch.get_default_dtype())
        features_train = np.stack(data['features_train'], axis=-1)
        features_test = np.stack(data['features_test'], axis=-1)
        if normalize:
            features_train = (features_train - train_norms[0]) / train_norms[1]
            # features_test = (features_test - test_norms[0]) / test_norms[1]
            features_test = (features_test - train_norms[0]) / train_norms[1]
        train_set = (features_train, data['u_train'])
        test_set = (features_test, data['u_test'])

        if(self.eval_PCA == True):
            pca = PCA(n_components=0.9999).fit(train_set[1])

            self.pca_basis      = torch.tensor(pca.components_.T, dtype=torch.get_default_dtype())
            self.pca_mean       = torch.tensor(pca.mean_, dtype=torch.get_default_dtype())
            
            self.num_pca_basis  = self.pca_basis.shape[1]

        self.train_set   = DONDataset(train_set)
        self.test_set    = DONDataset(test_set)

        sampler_train = torch.utils.data.sampler.BatchSampler(
                        torch.utils.data.sampler.RandomSampler(self.train_set),
                        batch_size=int(len(self.train_set)),
                        drop_last=False)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, sampler=sampler_train)

        # test sampler
        sampler_test =  torch.utils.data.sampler.BatchSampler(
                        torch.utils.data.sampler.RandomSampler(self.test_set),
                        batch_size=int(len(self.test_set)),
                        drop_last=False)


        self.test_loader = torch.utils.data.DataLoader(self.test_set, sampler=sampler_test)

class Helm3DDONDataset(object):
    def __init__(self, data_path=None, eval_PCA=False, batch_size=None):
        self.eval_PCA =  eval_PCA

        def bar_progress(current, total, width=80):
          progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
          sys.stdout.write("\r" + progress_message)
          sys.stdout.flush()


        if((data_path==None and os.path.isfile("NonNestedHelm3D_5000_1_32_1_0.0001.pkl")==False) or (data_path is not None and os.path.isfile(data_path)==False)):
            if(dist.get_rank()==0):
                print("Helm3DDONDataset:: Downloading dataset... ")
                url = 'https://zenodo.org/records/10904349/files/NonNestedHelm3D_5000_1_32_1_0.0001.pkl?download=1'
                
                if(data_path is not None):
                    filename = wget.download(url, data_path, bar=bar_progress)             
                else:
                    filename = wget.download(url, bar=bar_progress)       
        else:      
            print("Helm3DDONDataset:: Dataset found ")


        if(data_path==None):
            data_path="NonNestedHelm3D_5000_1_32_1_0.0001.pkl"


        data = pickle.load(open(data_path, 'rb'))
        self.permutation_indices = data['permutation_indices_don']
        self.reshape_feature_tensor = (-1, 1, 33, 33, 33)
        self.train_coordinate = torch.tensor(data['nodes_fem'], dtype=torch.get_default_dtype())
        
        features_train = data['features_don_train'][0].take(self.permutation_indices, axis=1).reshape(self.reshape_feature_tensor)
        features_test = data['features_don_test'][0].take(self.permutation_indices, axis=1).reshape(self.reshape_feature_tensor)
        train_set = (features_train, data['u_train'])
        test_set = (features_test, data['u_test'])

        if(self.eval_PCA == True):
            pca = PCA(n_components=0.9999).fit(train_set[1])

            self.pca_basis      = torch.tensor(pca.components_.T, dtype=torch.get_default_dtype())
            self.pca_mean       = torch.tensor(pca.mean_, dtype=torch.get_default_dtype())
            
            self.num_pca_basis  = self.pca_basis.shape[1]

        self.train_set   = DONDataset(train_set)
        self.test_set    = DONDataset(test_set)

        args = get_params()

        # TODO:: fix
        sampler_train = torch.utils.data.sampler.BatchSampler(
                        torch.utils.data.sampler.RandomSampler(self.train_set),
                        batch_size=int(len(self.train_set)/args.num_batches) if batch_size is None else batch_size,
                        drop_last=False)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set, sampler=sampler_train)

        # test sampler
        # TODO:: fix
        sampler_test =  torch.utils.data.sampler.BatchSampler(
                        torch.utils.data.sampler.RandomSampler(self.test_set),
                        batch_size=int(len(self.test_set)) if batch_size is None else batch_size,
                        drop_last=False)


        self.test_loader = torch.utils.data.DataLoader(self.test_set, sampler=sampler_test)


