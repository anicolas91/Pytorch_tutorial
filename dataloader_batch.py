'''
definitions:
epoch = 1 forward and backward pass of ALL training samples
batch_size = number of training samples in one fwd and bwd pass
number of iterations = number of passes, each pass using [batch_size] number of samples

eg 100 samples, batch size = 20 ---> 100/20 = 5 iterations for 1 epoch
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# we create our own custom dataset
class WineDataset(Dataset):
    # implement init data loading
    def __init__(self):
        #data_loading
        # data from wine.csv which is on https://github.com/patrickloeber/pytorchTutorial/
        # the data has 178 rows and 15 columns, 
        # sofor 178 samples it has 14 input parameters and 1 response
        xy = np.loadtxt('wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[:,1:]) # columns 1 onwards are all the input vars
        self.y = torch.from_numpy(xy[:,[0]]) #first column is the response
        self.n_samples = xy.shape[0] # no of rows

    # allow indexing
    def __getitem__(self,index):
        #dataset[0]
        return self.x[index], self.y[index]
    
    # allow length method
    def __len__(self):
        #len(dataset)
        return self.n_samples
    
# implement dataset
dataset = WineDataset()
# unpack first row to see whats inside for funsies
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# create a data loader and shuffle data for training purposes
# by setting num workers you use multiple processers and it loads faster, set to 0 if error
batch_size = 4
dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

# use an iterator to look into the data
dataiter = iter(dataloader) # iter creates an iterator across all items in the data
data = next(dataiter)
features, labels = data #unpack the data
print(features,labels)

# set up number of epochs and iterations needed for training
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/batch_size)
print(total_samples,n_iterations) # 178 samples, and 45 iterations

# create a training loop
# go through epocs
for epoch in range(num_epochs):
    # go through the train loader
    for i, (inputs,labels) in enumerate(dataloader):
        # forward
        # backward
        # update weight
        # print info
        if (i+1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')


# fun fact, pytorch has already some datasets inside, like mnist or fashion stuff
# the MNIST database is data of handwritten digits with a training set of 60,000 examples,
# and a test set of 10,000 examples. They have been normalized and centered
# to obtain it is: torchvision.datasets.MNIST()