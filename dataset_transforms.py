'''
pytorch has a lot of transforms for us:
# https://pytorch.org/vision/0.9/transforms.html


Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)]) #multiple transforms after each other

'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# we create our own custom dataset
class WineDataset(Dataset):
    # implement init data loading and optional transform
    def __init__(self,transform=None):
        # data from wine.csv which is on https://github.com/patrickloeber/pytorchTutorial/
        xy = np.loadtxt('wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        # we do not change things to tensor yet because we are going to use a transform for that instead
        self.x = xy[:,1:] # columns 1 onwards are all the input vars
        self.y = xy[:,[0]] #first column is the response
        self.n_samples = xy.shape[0] # no of rows
        self.transform = transform

    # allow indexing as dataset[0]
    def __getitem__(self,index):
        sample = self.x[index], self.y[index]

        # if transform is given apply it
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    # allow length method like len(dataset)
    def __len__(self):
        return self.n_samples
    
# create a transform class to convert things to tensors
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
        
# create a transform that multiplies things
class MulTransform:
    def __init__(self,factor):
        self.factor = factor
    
    def __call__(self,sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

# implement dataset with a transform
dataset = WineDataset(transform=ToTensor())

# check that it worked
first_data = dataset[0]
features, labels = first_data
print(type(features),type(labels))

# what happens if you do not apply any transform
dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print('no transform...')
print(type(features),type(labels))
print(features,labels)

# try a composed transform to convert to tensor and then multiply inputs by 2
composed = torchvision.transforms.Compose([ToTensor(),MulTransform(2)]) #input of compose must be a list
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print('composed transform...')
print(type(features),type(labels))
print(features,labels)
