# we will use MNIST dataset
# 1. we will implement dataloader and transformation (data preprocessing)
# 2. multilayer neural net, activations functions
# 3. we will set up the loss and the optimizer
# 4. we will batch train the training loop
# 5. we will evaluate the model
# opt: we will use GPU if thats available

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# set up device config... but mac has no cuda so its moot, just for fun
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up hyper parameters
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# 1. import mnist data (handwritten letters) using dataloader and transforms
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,
                                           transform=transforms.ToTensor(),download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',train=False,
                                           transform=transforms.ToTensor(),download=False)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,
                         shuffle=False) # no need to shuffle test data

# get samples and labels
examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape) # 100, 1, 28, 28 and 100 labels

# plot a couple of them, images are 28x28 pixels
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap = 'gray')
plt.show()


# 2. generate a neural net class with a hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU() # activation fcn
        self.l2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.l1(x) #linearize
        out = self.relu(out) # activation fcn
        out = self.l2(out) #linearize
        
        # no softmax here because the loss does that automatically
        return out

model = NeuralNet(input_size,hidden_size,num_classes).to(device)


# 3. loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate) #dw

# 4. create the training loop with batches
n_total_steps = len(train_loader) #number of batches
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader): #training data loader
        # reshape images into 1D string of sorts
        # original shape: 100, 1, 28, 28
        # reshaped: 100, 784
        images = images.reshape(-1,28*28).to(device) # send to  GPU/CPU
        # send labels to device
        labels = labels.to(device)

        # apply forward
        outputs = model(images) # give input and create predicted output
        loss = criterion(outputs,labels) # loss given predicted vs actual

        # apply backward
        loss.backward()
        optimizer.step()

        # ensure gradient reset
        optimizer.zero_grad()

        # print out the loss
        if (i+1) %100 == 0 :
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# 5. Evaluate the model via testing (this one is fast)
with torch.no_grad(): # we do not want to compute the gradient for all the steps anymore
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        # reshape images into 1D string of sorts
        images = images.reshape(-1,28*28).to(device) # send to  GPU/CPU
        # send labels to device
        labels = labels.to(device)
        # get test predictions
        outputs = model(images)
        # check how many predictions matched actual labels
        # value, index (class label)
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0] # adding how many samples were in that batch
        n_correct += (predictions == labels).sum().item() # get all predictions matching actual

# check the accuracy
acc = 100.0 * n_correct/n_samples
print(f'accuracy = {acc}')