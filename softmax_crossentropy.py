'''
softmax: S = e^y/sum(e^y)   -> it squashes output into 0 and 1
softmax essentially looks like a CDF, its a normalized exp fcn
SOFTMAX:
it converts scores into probabilities, with the highest score having the most probability

Softmax gets combined with the entropy loss to measure the performance of the classification model
cross entropy is a value btw 0 and 1
The better our prediction the lower our loss
loss entropy D(y_hat,y) = -1/n * sum(Y * log(Y_hat) ) 
Y needs to be one-hot encoded labels (binary 1/0)
Y_hat needs to be probabilities from softmax
'''

import torch
import torch.nn as nn 
import numpy as np 

# softmax description
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy: ',outputs) # highest probability for 2.0 -> 0.66
# 0.7 0.2, 0.1

# can also use pytorch tensor
x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x,dim=0)
print('softmax pytorch:', outputs) # highest prob 0.66

# cross entropu definition numpy
def cross_entropy(actual,predicted):
    loss = - np.sum(actual*np.log(predicted))
    return loss # / float(predicted.shape[0]) # use this to normalize

# y must be one hot encoded
# if class 0 : [1 0 0]
# if class 1 : [0 1 0]
# if class 2 : [0 0 1]

Y = np.array([1,0,0])

# y predicted is probabilities
Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad  = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}') # good 0.35 low loss
print(f'Loss2 numpy: {l2:.4f}') # bad 2.3 high loss

# cross entropy pytorch
# it already applise the softmax so no need to apply ourselves
# y prediction is raw too, no soft max probabilities
loss = nn.CrossEntropyLoss()

Y = torch.tensor([0]) # class 0 : [1,0,0], just need the raw class
# nsamples x nclasses = 1 x 3
Y_pred_good = torch.tensor([[2.0,1.0,0.1]]) # raw no softmax
Y_pred_bad  = torch.tensor([[0.5,2.0,0.3]]) # raw no softmax

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)
print(f'Loss1 torch: {l1.item():.4f}') # good 0.42 low loss
print(f'Loss2 torch: {l2.item():.4f}') # bad 1.8 high loss

# to get actual predictions:
_, predictions1 = torch.max(Y_pred_good,1)
_, predictions2 = torch.max(Y_pred_bad,1)
print(predictions1) # 0 --> correct
print(predictions2) # 1 ---> incorrect

# torch also allows for several class labels
Y = torch.tensor([2,0,1]) # just need the raw class
# nsamples x nclasses = 3 x 3
Y_pred_good = torch.tensor([[0.1,1.0,2.1],[2.0,1.0,0.1],[0.1,3.0,0.1]]) # raw no softmax
Y_pred_bad  = torch.tensor([[2.1,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]]) # raw no softmax

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)
print('3 classes...')
print(f'Loss1 torch: {l1.item():.4f}') # good 0.3 low loss
print(f'Loss2 torch: {l2.item():.4f}') # bad 1.6 high loss
print(Y_pred_good)
print(Y_pred_bad)
# to get actual predictions:
_, predictions1 = torch.max(Y_pred_good,1)
_, predictions2 = torch.max(Y_pred_bad,1)
print(predictions1) # 2,0,1 ---> correct
print(predictions2) # 0,2,1 ---> incorrect


## Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size,hidden_size,num_classes):
        super(NeuralNet2,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU() # rectified lienar unit
        self.linear2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.linear1(x) #linear function to hidden
        out = self.relu(out) # rectification
        out = self.linear2(out) # linear fcn to classes
        # no softmax at the end
        return out
    
# create model for 3 clases, dog cat none
model = NeuralNet2(input_size = 28*28, hidden_size=5,num_classes=3)
criterion = nn.CrossEntropyLoss() #applies softmax

## Binary problem
class NeuralNet1(nn.Module):
    def __init__(self, input_size,hidden_size): # no need for num clases since there are only 2
        super(NeuralNet1,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU() # rectified lienar unit
        self.linear2 = nn.Linear(hidden_size,1) # 1 class only

    def forward(self,x):
        out = self.linear1(x) #linear function to hidden
        out = self.relu(out) # rectification
        out = self.linear2(out) # linear fcn to classes
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred
    
# create model for 2 classes, dog or not dog
model = NeuralNet1(input_size = 28*28, hidden_size=5)
criterion = nn.BCELoss() # sigmoid, binary cross entropy loss
