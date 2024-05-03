# we will manually do the following:
# 1. Prediction
# 2. gradients computation
# 3. loss computation
# 4. parameter updates

# once we do it manually we can automate via:
# 1. Pytorch model
# 2. autograd
# 3. pytorch loss
# 4. pytorch optimizer

import numpy as np 

# we use linear regression f = w*x (we dont care about bias +b)

# f = 2*x

X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    return w*x

# loss
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean() #MSE

# gradient
# MSE = 1/N * (wx-y)^2
#dJ/dw = 1/N * 2x * (wx-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5)}.3f')

# training
learning_rate = 0.01
n_iters = 10

# loop
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y,y_pred)
    # gradients
    dw = gradient(X,Y,y_pred)
    # update weights
    w -= learning_rate*dw # gradient descent algo

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss: {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
