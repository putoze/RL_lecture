#
# Simple example to demonstrate PyTorch variables, expressions, gradients
#

import torch

a=torch.tensor([1.,2.,3.],requires_grad=True)
b=torch.tensor([4.,5.,6.],requires_grad=True)

def compute(x):
    x=torch.tensor(x,dtype=torch.float32)
    output=torch.sum(a**2*x+b)
    return output

result=compute([7.,8.,9.])
result.backward() #compute gradients on model graph

#print gradients & result
print(a.grad)  #grad wrt a is 2ax
print(b.grad)  #grad wrt b is 1
print(result)
print('----------\n')

#There are some situations where we may need or want to 
# suppress automatic gradient tracking. This can
# be achieved using the no_grad() function, as follows
#
with torch.no_grad():
    result=compute([10.,11.,12.])

print(result)
print('----------\n')
#result.backward() #note, this won't work here because gradients were not computed!

#
#Can easily convert tensors to & from numpy arrays.
#
c=torch.tensor([9.,10.,11.])
d=torch.tensor([12.,13.,14.], requires_grad=True)

print(c.numpy())
print(d.detach().numpy()) #if requires_grad=True, must first detach

import numpy as np
e=np.array([15.,16.,17.],dtype=np.float32)
print(e)
f=torch.from_numpy(e)
print(f)
print('----------')

