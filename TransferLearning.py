'''
Transfer learning is basically using a pretrained model, 
changing the last layer on the training bit, and using a new
classification system.
Its cheaper and faster to reuse than to make a new model from 
scratch.

/in this example we will use a pretrained network trained on
a million images, is 18 layers deep and classifies 1000 obj
categories.

We only will have 2 categories, ants and bees

# we will use ImageFolder, Scheduler, n Transfer Learning
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler # to find a good lr isntead of just winging it
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
import time
import os
import copy
from torch.utils.data import DataLoader


# setup what device to send  to
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet's mean and std on rgb
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#set up transforn for images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

#1. import the data
data_dir = 'data/hymenoptera_data'
# use generic data loader imageFolder to extract images 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                  for x in ['train', 'val']}
# load the data for train and evaluation
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=0)
              for x in ['train', 'val']}
# get the sizes of each train and val dataset
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# get all the classes names (bee and ant)
class_names = image_datasets['train'].classes
#print the class names
print(class_names)

# 2. create training model fcn
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() #start clock

    best_model_wts = copy.deepcopy(model.state_dict()) #copy existing model net weights
    best_acc = 0.0 #start the accuracy var

    # loop across epochs (default is 25)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0 #start calc var for loss
            running_corrects = 0 #start calc var for correct count

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad() #reset grads
                        loss.backward() # calculate dw
                        optimizer.step() #update step w

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step() # scheduler will update and improve learning rate

            # check loss and accuracy at each epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model if the accuracy of the epoch is better than previous
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since # stop the clock
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

## 1) FINETUNING ==============================================================
# we are training the whole model again, but finetuning the weights 
# a little bit with the new last layer
print('finetuning...')
# import pretrained model
model = models.resnet18(pretrained=True)

#exchange the last fully connected layer
num_ftrs = model.fc.in_features
num_classes = 2 # only two, bee and ant
model.fc = nn.Linear(num_ftrs,num_classes)# create new layer and assign to last
model.to(device)

# set up loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001) # stochastic gradient

# set up scheduler, will update the learning rate
# every 7 epochs the learning rate will be multiplied by gamma
step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

# call the training function made above
nodel = train_model(model,criterion,optimizer,num_epochs=20)

## 2) FREEZING ==============================================================
# freeze all layers in the beginning and only train new last layer
## much faster!!
print('freezing all but last...')
model = models.resnet18(pretrained=True)
# freeze layers
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
num_classes = 2 # only two, bee and ant
model.fc = nn.Linear(num_ftrs,num_classes)# create new layer and assign to last
model.to(device)
# set up loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001) # stochastic gradient
# set up scheduler, will update the learning rate
step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)
# call the training function made above
nodel = train_model(model,criterion,optimizer,num_epochs=20)