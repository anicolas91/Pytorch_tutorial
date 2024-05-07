'''
we will classify the CIFAR dataset (has animals, and methods of transport)
we will use a convolutional neural networks
CNN mainly work on images using neurons w weight and biases
you use convolutional filters
we will have different convolutional layers followed by some activation layers, with some POOL
layers in between.
pooling layers automatically learn features from images
at the end we have fully connected layers for actual classification task
and they will give the likelihood that it is one particular class

What does a convolutional layer do:
We use a kernel on each pixel as a filter to create a resulting image 
The resulting layer will have smaller images

what is max pooling layer?
this layer downsamples the image by getting the max pixel value in the specified region
for example if we pool with a 2x2 we grab 4 pixels and get their max
it helps with speed and reduces overfiting


'''

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 
from torch.utils.data import DataLoader

# set up device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up hyperparameters
num_epochs = 4 # accuracy is 50% may need more epochs and better learning rate probs
batch_size = 4
learning_rate = 0.001

# set up transform bit
meanrgb = (0.5,0.5,0.5) # mean value for each rgb channel
stdrgb = (0.5,0.5,0.5) # standard dev for each rgb channel
# if mean and stdev is 0.5, then the range is -1 to 1
# image = (image-mean)/stdev
#min when image ==0 --> (0-0.5)/0.5 = -1
#max when image ==1 --> (1-0.5)/0.5 =  1
#max = 0.5+1.5 = 1
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(meanrgb,stdrgb)])

# 1. import CIFAR10 data (60k 32x32 images in 10 classes) using dataloader and transforms
train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True,
                                           transform=transform,download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                           transform=transform,download=False)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,
                         shuffle=False) # no need to shuffle test data

# 2. hardcode classes
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#3. generate a cnn class model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        # conv relu -- pool -- conv relu -- pool-- full connectx3 w relus
        rgb_channels = 3 # red yellow blue
        out_channel1 = 6
        out_channel2 = 16
        num_classes = 10
        flatten_size = 16*5*5 # final size of tensor flattened
        # size is [4,16,5,5] after conv1--pool--conv2--pool passes
        # to calculate post size: (W-F+2P)/S + 1
        # W = width, F = filter size, P = padding, S = stride
        #             W ,F,P,S
        # after conv1 32,5,0,1==> (32-5+0)/1+1 = 28
        # after pool  28,2,0,2==> (28-2+0)/2+1 = 14
        # after conv2 14,5,0,1==> (14-5+0)/1+1 = 10
        # after pool  10,2,0,2==> (10-2+0)/2+1 = 5
        # so the input to flatten is 3,16,5,5
        self.conv1 = nn.Conv2d(rgb_channels,out_channel1,kernel_size=5)
        self.pool = nn.MaxPool2d(2,2) # kernel size and stride
        self.conv2 = nn.Conv2d(out_channel1,out_channel2,kernel_size=5)
        self.fc1 = nn.Linear(flatten_size,120) # fully connected layer need 3
        self.fc2 = nn.Linear(120,84) # can change 120 and 84
        self.fc3 = nn.Linear(84,num_classes) # final layer
        #softmax not needed, already on cross entropy loss

    def forward(self,x):
        # first conv relu
        x = F.relu(self.conv1(x))
        # pool
        x = self.pool(x)
        # second conv relu
        x = F.relu(self.conv2(x))
        # pool
        x = self.pool(x)
        # flatten tensor
        x = x.view(-1,16*5*5) #3,400
        # do fully connected relu 3 passes (1 per channel)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # final bit doesnt need activation fcn
        # no need for softmax, already on loss
        return x


# setup model
model = ConvNet().to(device) # remember to send model, and inputs to device

#4. setup loss and optimizer
criterion = nn.CrossEntropyLoss() #evaluates loss with probability vals
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate) # stoch grad descent

# 5. create training loop with batches
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        #reshape 2d images into 1d
        # original shape: 4, 3, 32, 32
        # new shape: 4, 3, 1024

        # send inputs to device
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images) # predict class
        loss = criterion(outputs,labels) # calculate loss pred vs actual
        
        # backward pass
        optimizer.zero_grad() # ensure gradient doesnt accumulate
        loss.backward() # calculate dw
        optimizer.step() # go to the next step

        # print out the loss
        if (i+1) % 1000 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')

print('finished training.')

# 6. test it out and check for accuracy
with torch.no_grad(): # no need to calculate/update gradients 
    # initialize params
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    # start looping across batches
    for images, labels in test_loader:
        # send inputs to device
        images = images.to(device)
        labels = labels.to(device)
        # predict class
        outputs = model(images)
        # max returns (value, index), just get max of prediction in vector
        _, predicted = torch.max(outputs,1)
        # add up the number of correct predictions
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        # add up per class the number of correctp predictions
        for i in range(batch_size): # go 1 by 1 in that batch of the test loader
            label = labels[i] # gets label idx
            pred = predicted[i] # gets pred idx
            if label == pred:
                n_class_correct[label] +=1 # add count on that label idx
            n_class_samples[label] +=1 #just count another one for that label

    # calculate the accuracy
    acc = 100.0*n_correct/n_samples
    print(f'accuracy of CNN: {acc} %')

    # check accuracy per class
    for i in range(10):
        acc = 100 * n_class_correct[i]/n_class_samples[i]
        print(f'accuracy of {classes[i]}: {acc} %')