#this is all gradient calc in pytorch
import torch

#lets say you got F = a*b where
# a = 10
# b = 20
# so dF/da = b = 20
# and dF/db = a = 10

# in pytorch this looks like:
a = torch.tensor(10.0,requires_grad=True) # floating and complex only
b = torch.tensor(20.0,requires_grad=True)
F = a*b # the function is established

# to calculate the gradients...
F.backward()
print(F) # gradients saved

# to see them
print(a.grad) # 200/10
print(b.grad) # 200/20

# for vectors , it doesn't work like this anymore. when we call
# the backward function to the tensor if the tensor is 
# non-scalar (i.e. its data has more than one element) and
# requires gradient, the function additionally requires 
# specifying gradient.

a = torch.tensor([10.0,10.0],requires_grad=True)
b = torch.tensor([20.0,20.0],requires_grad=True)

F = a*b
v = torch.tensor([1.0,1.0])
F.backward(gradient=v)

# were doign a full jacobian vector product J*v
# so the jacobian * the vector
# remember J = grad y / grad x
# f1 = a1*b1
# f2 = a2*b2
# for df/da: 
# df1/da1 df1/da2  @  1  =>  b1   0   @ 1  ==> b1  b2 --> 20 20
# df2/da1 df2/da2     1      0    b2    1
# for df/db: 
# df1/db1 df1/db2  @  1  =>  a1   0   @ 1  ==> a1  a2 --> 10 10
# df2/db1 df2/db2     1      0    a2    1

# chain rule for F.grad
# for df/df = 1 ? 
# df1/da1/db1 df1/da2/db2  @  1  =>  b1/a1   0   @ 1  ==> b1  b2 --> 20 20
# df2/da1/db1 df2/da2/db2     1      0    b2    1

print(a.grad) # gradient df/da
print(b.grad) # gradient df/db
print(F)
print(F.grad)

# you can also propagate
a = torch.tensor([10.0,10.0],requires_grad=True)
b = torch.tensor([20.0,20.0],requires_grad=True)
F = a*b # the function is established
# to preserve gradients before more calcs, retain grad
F.retain_grad()
G = 2*F
G.backward(gradient=torch.tensor([1.0,1.0]))
print(a.grad) #dG/da
print(b.grad) #dG/db
print(F.grad) # will show none if not retained grad because it is no longer a leaf in the calc tree

# for optimization you should make sure that gradients reset to zero
# with zero gradients
print('analyzing epochs...')
weights = torch.ones(4,requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    print(model_output) # 12
    model_output.backward()
    print(weights.grad) # F = 3w1 + 3w2 + 3w3 + 3w4 
                    #     J = df1/dw1, df2/dw2, df3/dw3, df4/dw4
                    #     J = 3, 3, 3, 3
    # unless you reset grad to zero, it accumulates gradient of w...
    # 3 ,3,3,3 + 3,3,3,3...
    # weights.grad.zero_()

# for optimizers this looks like:
optimizer = torch.optim.sgd(weights, lr = 0.01)
optimizer.step()
optimizer.zero_grad()

# tldr: 
#1. remember to set requires_grad to True to use backward
#2. to optimize you need to reset gradients


# to update weights without considering the gradient computation:
# 1. x.requires_grad_(False)
# 2. x.detach()
# 3. with torch.no_grad() 