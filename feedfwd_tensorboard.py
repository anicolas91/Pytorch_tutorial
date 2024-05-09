'''
IMPORTANT TENSORBOARD INFO!!
a) you may want to run this code first or create the folder 'runs'
b) having the folder 'runs' you can now run on terminal A:
    tensorboard --logdir=runs
The terminal will have tensorboard running on localhost:6006
c) anytime you want to see anything new on tensorboard:
    1. run on terminal B the current python script
    2. refresh the tensorboard website

'''
# project copied from MNIST feedforward efforts
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import sys
import torch.nn.functional as F
# Set up tensorboard writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/mnist2') # changing to mnist2 creates a secondary line on tensorboard

# set up device config... but mac has no cuda so its moot, just for fun
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up hyper parameters
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

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
#print(samples.shape, labels.shape) # 100, 1, 28, 28 and 100 labels

# plot a couple of them, images are 28x28 pixels
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap = 'gray')
#plt.show()
    
# TENSORBOARD =====================================
im_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images',im_grid)
# end it here for now
# writer.close()# make sure outputs are all flushed
# sys.exit()
# important note: to have tensorboard working 
# 0. start on a fresh terminal
# 1. save and run this script.
# 2. run on terminal 'tensorboard --logdir=runs
# the point is to have the existing directory 'runs' by the time this script runs.

# ================================================

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

# TENSORBOARD =====================================
writer.add_graph(model,samples.reshape(-1,28*28)) # model and input
# end it here for now
# writer.close()# make sure outputs are all flushed
# sys.exit()
# important note: to have tensorboard working 
# 0. start on a fresh terminal
# 1. save and run this script.
# 2. run on terminal 'tensorboard --logdir=runs
# the point is to have the existing directory 'runs' by the time this script runs.

# ================================================

# 3. loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate) #dw

# 4. create the training loop with batches
n_total_steps = len(train_loader) #number of batches

running_loss = 0
running_correct = 0
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

        # calculate running loss and correct
        running_loss += loss.item()
        _, predictions = torch.max(outputs.data,1)
        running_correct += (predictions == labels).sum().item() # get all predictions matching actual

        # print out the loss
        if (i+1) %100 == 0 :
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            # TENSORBOARD ==============================================================================
            writer.add_scalar('training loss',running_loss/100,global_step=epoch*n_total_steps+i)
            writer.add_scalar('accuracy',running_correct/100,global_step=epoch*n_total_steps+i)
            #reset training loss and accuracy very 100 steps
            running_loss = 0.0
            running_correct = 0
            # =========================================================================================

# 5. Evaluate the model via testing (this one is fast)

#initiate lists
labels_print = []
preds_print = []
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

        #use softmax activation fcn to calculate probabilities 
        class_predictions = [F.softmax(output,dim=0) for output in outputs]
        #add to lists
        labels_print.append(labels)
        preds_print.append(class_predictions)

    # concatenate everything
    labels_print = torch.cat(labels_print) # 10,000x1
    preds_print = torch.cat([torch.stack(batch) for batch in preds_print]) #10,000x10

    # create class labels for plotting
    classes = range(10) #numbers 0 to 9
    #TENSORBOARD ========================================================
    for i in classes:
        labels_i = labels_print == i
        preds_i = preds_print[:,i]
        writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
        writer.close()
    # ===================================================================
# check the accuracy
acc = 100.0 * n_correct/n_samples
print(f'accuracy = {acc}')