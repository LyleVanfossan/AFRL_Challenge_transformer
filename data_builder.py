import torch
import numpy as np
from torch.utils import data

class data_builder():
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, transforms = None):
        'Initialization'
        self.transforms = transforms
        self.inps = data[:-1]
        self.target = data[1:]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.inps)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.inps[index]
        y = self.target[index]
        return X, y

'''
# 

# datasets
d = np.linspace(1, 100, 100)
custom_dataset = data_builder(d)

# dictionary of network parameters
params = {
    'batch_size': 5,
    'shuffle': False
}

train = data.DataLoader(custom_dataset, **params)

for b, bl in train:
    if len(b) == params['batch_size']:
        print(b)
        print(bl)
        print("\n")
    else:
        continue
'''