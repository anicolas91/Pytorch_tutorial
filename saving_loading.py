'''
there are two ways to save models:
# COMPLETE MODEL (LAZY METHOD) =========================
torch.save(model,PATH)

TO LOAD THE MODEL:
# model class must be defined somewhere
model = torch.load(PATH)
model.eval()

SERIALIZED DATA IS BOUND TO CLASSES AND STRUCTURE

# SAVE TRAINED MODEL ONLY (PREFERRED) ====================
torch.save(model.state_dict(),PATH)

TO LOAD THE MODEL:
# model needs to be created again with ł
model = Model(*args,**kargs)
model.load_state_dict(torch.load(PATH))
model.eval()

# TO SAVE TO GPU AND LOAD ON CPU =========================
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(),PATH)

device = torch.device('cpu')
model = Model(*args,**kargs)
model.load_state_dict(torch.load(PATH,map_location=device))

# TO SAVE TO GPU AND LOAD ON GPU =============================
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(),PATH)

model = Model(*args,**kargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# TO SAVE TO CPU AND LOAD ON GPU ===============================
torch.save(model.state_dict(),PATH)

device = torch.device('cuda')
model = Model(*args,**kargs)
model.load_state_dict(torch.load(PATH,map_location='cuda:0')) #choose gpu num
model.to(device)
'''

import torch
import torch.nn as nn
import sys

# create sample model
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model,self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model(n_input_features=6)
#train your model...
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# can also save checkpoints via a dictionary
checkpoint = {
    'epoch':90,
    'model_state': model.state_dict(),
    'optim_state': optimizer.state_dict()
}

# will need to save the checkpoint itself
# torch.save(checkpoint,'checkpoint.pth')

# then you can load the checkpoint and access the vars
loaded_checkpoint = torch.load('checkpoint.pth')
epoch = loaded_checkpoint['epoch']
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(),lr=0) # if we load the good state lr will change

# load model and optimizer and access checkpoint
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])

# print out the state dict of model (weights and bias tensor) 
# and optimizer (momentum and learning rate)
print('model state:',model.state_dict())
print('optimizer state:',optimizer.state_dict())
sys.exit()

# saving the model...
FILE = 'model.pth'
#  LAZY METHOD ===============================
# SAVE
#torch.save(model,FILE)
# output is some serialized data 

# LOAD
# model = torch.load(FILE)
# model.eval()

# when loading model you have the parameters too
for param in model.parameters():
    print(param)

# STATE DICT METHOD (RECOMMENDED)=========================
# SAVE
torch.save(model.state_dict(),FILE) #saves binary data
# LOAD
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

# it preserves the parameters from orig
for param in loaded_model.parameters():
    print(param)

