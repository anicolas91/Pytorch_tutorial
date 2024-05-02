import torch
import numpy as np 
# general notes: torch is a lot like numpy, except that it works with tensors really

# create a tensor of n dimensions
x = torch.empty(2,3,3)
print(x)

# create zeros and random numbers
x = torch.rand(3,2)
print(x)

x = torch.zeros(2,2)
print(x)

# can also set up data type
x = torch.ones(3,1,dtype=torch.float16)
print(x)
print(x.dtype)

# can check size
print(x.size())

# can create a tensor based on a list
x = torch.tensor([2.5,1.0])
print(x)

# can do math duh
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x + y

print(x)
print(y)
print(z)

# other inplace operations possible
y.add_(x)
print(y)

# symbols can be replaced by torch commands
z = x / y
print(z)
z = torch.div(x,y)
print(z)

z = x * y
print(z)
z = torch.mul(x,y)
print(z)

# just like numpy you can slice
x = torch.rand(5,3)
print(x)
print(x[:,0])
print(x[1,:])

#we can use item to get the actual value inside a 1,1 tensor
print(x[1,1].item())

# we can reshape the tensor
x = torch.rand(4,4)
print(x)
y = x.view(16) # converts to 1D
print(y)

# if you want to specify 1 side of dimesion, and have pytorch figure out the other dims, then use -1
y = x.view(-1,8)
print(y)
print(y.size())

# we can convert btw numpy and torch
a = torch.ones(5)
b = a.numpy()
print(type(b)) # this is a numpy array

# if you work on a cpu and not a gpu, then one change apples to both

a.add_(1)
print(a) 
print(b) #  this also gets a +1 because it shares the asme memory location


# inversely
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a +=1
print(a)
print(b) # tensor got modified too, same memory

# you can do operations on GPU but for that you need the CUDA toolkit
# mac doesnt have this
print(torch.cuda.is_available())

# if you had cuda you could do:
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y # both are on GPU so its good
    # need to come back to CPU to use numpy
    z = z.to("cpu")

# when tensor is created, you see requires_grad
# it tells torch that youll need to calc gradients for this one, likely for optimization purposes
x = torch.ones(5,requires_grad=True)
print(x) # so this will have the gradient flag as true for later use