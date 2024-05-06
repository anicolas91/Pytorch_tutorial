'''
Popular activation functions are:
1. Step fcn (not used in practice) --> just a step on/off
2. Sigmoid -->value btw 0 and 1, usually in last layer of binary classification problem
3. TanH -> scaled sigmoid, used on hidden layers (val btw -1 and +1)
4. ReLU --> split linear, 0 for neg and x=y otherwise, if dont know what to use, use this for hidden layers
5. Leaky ReLU --> slight gradient for neg values, x=y otherwise, fixes vanishing gradient problem
6. softmax --> good in last layer for multi class problems, it gives a probability btw 0 and 1

Vanishing gradient 'kills' the neurons because gradient is zero and they never update
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F 

# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()
        self.relu    = nn.ReLU()
        self.leakyrlu= nn.LeakyReLU()
        self.sotfmax = nn.Softmax()
        self.linear2 = nn.Linear(hidden_size,1)
    def forward(self,x):
        out = self.linear1(x) # linear transformation input to hidden
        out = self.relu(out) #hidden layer, relu, leaky relu, tanH
        out = self.linear2(out) #linear transformation hidden to output
        out = self.sigmoid(out) # final layer, sigmoid, softmax
        return out
    
# option 2 (use activation functions directly in fwd pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size,hidden_size):
        super().__init__(NeuralNet2,self)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,1)
    def forward(self,x):
        #torch.softmax
        #torch.tanh
        #sometimes they are not available in torch, like leaky relu
        # but they are available on the functional
        # F.leaky_relu
        # F.relu
        # F.tanh
        # F.sigmoid
        # F.softmax
        out = torch.sigmoid(self.linear2(out))
        return out