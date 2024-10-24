import torch
import requests
from torch.utils.data import random_split, Dataset, DataLoader

def download_shakespeare():
    url = 'https://github.com/jcjohnson/torch-rnn/blob/master/data/tiny-shakespeare.txt'  # tiny dataset for testing
    response = requests.get(url)
    data = response.text
    return data

def load_shakespeare(train_split=0.8, block_size=128):
    data = download_shakespeare()
    tokenizer = CharTokenizer(data)
    dataset = ShakespeareDataset(data, tokenizer, block_size=block_size)

    # Split dataset into training and testing
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset, tokenizer


# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, data):
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        return ''.join([self.itos.get(i, '') for i in l])

# Custom dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = data
        self.tokenized_data = tokenizer.encode(data)

    def __len__(self):
        return len(self.tokenized_data) - self.block_size

    def __getitem__(self, idx):
        x = self.tokenized_data[idx:idx+self.block_size]
        y = self.tokenized_data[idx+1:idx+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
