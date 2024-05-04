# general pipeline:
# steps
# 1. design model (input, output size and forward pass)
# 2. construct loss and optimizer
# 3. training loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

# 0. prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32)) # converts to tensors
Y = torch.from_numpy(Y_numpy.astype(np.float32))

print(X.shape) # this one is already 2D 100x1
print(Y.shape)
Y = Y.view(Y.shape[0],1) # reshapes as 2D size of 100 x 1

n_samples, n_features = X.shape # 100,1

# 1. model
input_size = n_features # 1
output_size= 1
model = nn.Linear(input_size,output_size)

# 2. loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # will calculate mean squared error
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) #model parameters has [w,b] from y = wx+b

# 3. training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    Y_predicted = model(X)
    loss = criterion(Y_predicted,Y)

    # backward pass
    loss.backward()

    # weight update
    optimizer.step()
    optimizer.zero_grad() #remember to empty

    # print info
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy,Y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()