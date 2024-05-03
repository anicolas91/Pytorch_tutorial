import torch

# we do fwd and backward
#x
#   *  -> y_hat
#w                -   ->  s ---> ^2 ---> loss
#           y           

# x = w =1, y = 2
#fwd : x*w -> y_hat = 1 --> -y --> s = -1 --> loss=s^2 = 1
# bwd : dloss/dw = dloss/ds*ds/dy_hat*dy_hat/dw
# dloss/ds = 2s
# ds/dy_hat = 1
# dy_hat/dw = x
# bwd: 2s*1*x = 2*(-1)*1*1 ===> -2

# in torch this looks like:

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0,requires_grad=True)

# forward adn compute loss
y_hat = (x*w)
loss = (y_hat-y)**2 # MSE really

print(loss)

# backward pass
loss.backward()
print(w.grad)

# for general ML you update the weights, and repeat fwd, bwd