# general pipeline:
# steps
# 1. design model (input, output size and forward pass)
# 2. construct loss and optimizer
# 3. training loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

# for logistic we will be use a different loss fcn

import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler # we will standardize/scale features
from sklearn.model_selection import train_test_split # we will split data into trainign and testing

# 0. prepare data
bc = datasets.load_breast_cancer() #binary classification problem
X,Y = bc.data,bc.target

n_samples, n_features = X.shape
print(n_samples,n_features)

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234) # 20%, using same state 1234

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test) # we used the fit from train

# convert to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test  = torch.from_numpy(Y_test.astype(np.float32))

# reshape as 2D, remember all Xs are already 2D
Y_train = Y_train.view(Y_train.shape[0],1)
Y_test  = Y_test.view(Y_test.shape[0],1)
print(Y_train.shape)
print(Y_test.shape)

# 1. model
# this model is y = wx+b, with a sigmoid at the end

class LogisticRegression(nn.Module): # this class is inheriting from nn.Module
    def __init__(self, n_input_features):
        super(LogisticRegression,self).__init__() #it gives access to methods and properties on the parent class nn.Module
        self.linear = nn.Linear(n_input_features,1) # only 1 output size

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features) #30 input and 1 output

# 2. loss and optimizer
learning_rate = 0.3 # was 0.01 originally, this got better
criterion = nn.BCELoss() # binary cross entropy loss, uses logs
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# 3. training loop
num_epochs = 100 # no significant improvement w increase

for epoch in range(num_epochs):
    # forward and loss
    Y_predicted = model(X_train)
    loss = criterion(Y_predicted,Y_train)

    #backward
    loss.backward()

    # update weights
    optimizer.step()

    # reset gradients to zero
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')

# evaluate model with training
with  torch.no_grad(): # do not keep the history
    # remember that the breast cancer data is just binary yes/no or 0/1
    Y_predicted = model(X_test)
    Y_predicted_cls = Y_predicted.round() # we just binarize 0/1
    total_predicted = Y_predicted_cls.eq(Y_test).sum() # how many true positive
    accuracy = total_predicted/float(Y_test.shape[0])
    print(f'accuracy: {accuracy:.4f}')

