import torch

#f = 2*x
X = torch.tensor([1,2,3,4],dtype = torch.float32)
Y = torch.tensor([2,4,6,8],dtype = torch.float32)

# weights
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# model prediction
def forward(x):
    return w*x

# loss
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean() #MSE


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
n_iters = 80

# loop
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)
    # loss
    l = loss(Y,y_pred)
    # gradients = backward pass
    l.backward() # dl/dw ... back propagation is not as exact as numerical
    # update weights
    with torch.no_grad(): # weight update should not be part of gradient computation
        w -= learning_rate*w.grad #it updates w but does not mess with dw

    # reset zero gradients
    w.grad.zero_()

    if epoch % 5 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss: {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
