#
# Simple example to demo some of the features & behavior
#  of PyTorch autograd machinery
#

import torch
torch.manual_seed(789)

#Small MLP with 1 input, 3 outputs, 1 hidden layer
#
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1=torch.nn.Linear(1,4)
        self.relu1=torch.nn.ReLU()
        self.dense2=torch.nn.Linear(4,3)
 
    def forward(self,s):
        feature_input=torch.tensor(s,dtype=torch.float32)

        #forward propagate input through network layers
        output=self.dense1(feature_input)
        output=self.relu1(output)
        output=self.dense2(output)

        return output
    
nn=MLP()

#What if we want to propagate gradients for only
# a single output?  Turns out in PyTorch this is
# easy because the indexing operator is included
# in gradient tracking
nn_output=nn.forward([1.0])
nn_output_1=nn_output[1]  #select output 1
nn_output_1.backward()    #compute gradients for output 1

#let's print out the gradients and see what happened
# Notice the gradients on weights connected to the 0th and 2nd
# outputs are all zero
for p in nn.parameters():
    print(p.grad)
    