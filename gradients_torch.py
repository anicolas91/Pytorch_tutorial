# steps
# 1. design model (input, output size and forward pass)
# 2. construct loss and optimizer
# 3. training loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

import torch
import torch.nn as nn

#f = 2*x
# X = torch.tensor([1,2,3,4],dtype = torch.float32)
# Y = torch.tensor([2,4,6,8],dtype = torch.float32)
# for torch nn models the shape needs to be 2D, with n rows being no of samples
X = torch.tensor([[1],[2],[3],[4]],dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype = torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features
# weights
# w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
# model prediction
# def forward(x):
#     return w*x

# to make your own pytorch model:
class LinearRegression(nn.Module):

    def __init__(self, input_dim,output_dim):
        super(LinearRegression,self).__init__() #super constructor
        # define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)
    

model = LinearRegression(input_size,output_size)

#weights and model prediction on torch become:
#model = nn.Linear(input_size,output_size) # just need size of input and output
[w,b] = model.parameters()
# the linear model is 1 layer and given byu torch already

# loss
# def loss(y,y_predicted):
#     return ((y_predicted-y)**2).mean() #MSE

# on pytorch this is:
loss = nn.MSELoss()

# to test, you need to this time set up a specific variable
X_test = torch.tensor([5],dtype=torch.float32)
#print(f'Prediction before training: f(5) = {forward(5):.3f}')
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# training
learning_rate = 0.01
n_iters = 100

# you also need an optimizer, with weights as a list or just model parameters
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) # SGD = stochastic gradient descent

# loop
for epoch in range(n_iters):
    # prediction = forward pass
    # y_pred = forward(X)
    # can use pytorch model now
    y_pred = model(X)
    # loss
    l = loss(Y,y_pred)
    # gradients = backward pass
    l.backward() # dl/dw ... back propagation is not as exact as numerical
    # update weights
    # with torch.no_grad(): # weight update should not be part of gradient computation
    #     w -= learning_rate*w.grad #it updates w but does not mess with dw
    # with torch you just call the optimizer
    optimizer.step() # this does the gradient descent of -dw*lr

    # reset zero gradients
    # w.grad.zero_()
    # also need to reset gradients in the optimizer
    optimizer.zero_grad()

    if epoch % 5 == 0:
        [w, b] = model.parameters()
        # w is a tensor inside the model parmeters so get the liist of list
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss: {l:.8f}')

# print(f'Prediction after training: f(5) = {forward(5):.3f}')
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
