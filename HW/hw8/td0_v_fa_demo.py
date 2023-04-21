#
#TD(0) state-value function evaluation with function approximation demo
#  - Value function approximation is done using small multi-layer feedforward neural network
#  - Test using 1-D Mouse-Cheese example with slightly modified optimal policy
#
#  1. Compute v(s,w) using semi-gradient TD(0) algorithm
#  2. Compare with tabular TD(0) algorithm
#

from td0_v_fa import td0_v_fa
from td0_v import td0_v
from cat_and_mouse import Cat_and_Mouse
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.manual_seed(456)  #let's make things repeatable! (only affects PyTorch neural-network param initialization in this demo)


#Small multi-layer feedforward neural network model for v (PyTorch)
#
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1=torch.nn.Linear(1,8)
        self.relu1=torch.nn.ReLU()
        self.dense2=torch.nn.Linear(8,4)
        self.relu2=torch.nn.ReLU()
        self.dense3=torch.nn.Linear(4,1)
 
    def v(self,s):
        '''
        Compute value function v(s,w) by forward computation through MLP   
        '''
        feature_input=torch.tensor([s],dtype=torch.float32)

        #forward propagate input through network layers
        output=self.dense1(feature_input)
        output=self.relu1(output)
        output=self.dense2(output)
        output=self.relu2(output)
        output=self.dense3(output)

        return output

    @property
    def weights(self):
        '''
        Return model parameters
        '''
        return self.parameters()


#Setup the 1-D cat-mouse simulation environment & policy:
#
# - odd number of tiles (>= 7)
# - init mouse location is center tile
# - two cheese locations, one on each edge (left, right)
# - 1 sticky tile near left edge, 1 slippery tile near right edge
# - policy is modified optimal policy with uniform random action choice
#    at mouse initial location (to encourage full exploration of environment)
#
# Note: to modify the problem size, you only need to change the num_tiles
#       parameter below; e.g., num_tiles=7, num_tiles=17, num_tiles=27...
# 7,17,27
num_tiles=17 #1-D environment size

if num_tiles % 2 == 0 or num_tiles < 7:
    raise Exception('num_tiles must be an odd integer >= 7, got num_tiles={}'.format(num_tiles))

#mouse init, cheese, sticky and slippery tile locations
init_loc=int((num_tiles-1)/2)
cheese_locs=[[0,0],[0,num_tiles-1]]
sticky_locs=[[0,2]]
slippery_locs=[[0,num_tiles-3]]

#Instantiate the simulation environment instance.
# This is the 1-D Mouse-Cheese example from earlier lectures
cm=Cat_and_Mouse(rows=1,columns=num_tiles,mouseInitLoc=[0,init_loc], cheeseLocs=cheese_locs,stickyLocs=sticky_locs,slipperyLocs=slippery_locs)
#cm.render()

#Modified optimal policy (e.g., mouse init state policy is random, to encourage full exploration)
policy=np.zeros((cm.numStates,cm.numActions),dtype=np.float32)
for i in range(num_tiles):
    if i < init_loc:
        policy[i,1]=1.0 #go left
    elif i > init_loc:
        policy[i,0]=1.0 #go right
    else:
        policy[i,0]=policy[i,1]=0.5 #randomly choose left, right (mouse init location)



#################################################################
#
# 1. Compute v(s) using semi-gradient TD(0) with function approximation
#
#

#Instantiate the v(s,w) model
vmodel=MLP()

#Compute v(s,w) using TD(0) method w/ function approximation
td0_v_fa(cm,vmodel,policy,0.9,1e-2,5000,100)

print('Value function (function approximation):')
#compute & plot v for all states, excluding absorbing terminal states
v_fa=[]
for s in range(1,cm.numStates-1):
    v_fa.append(float(vmodel.v(s)))
print(v_fa)
plt.plot(v_fa)
plt.show()

cm.reset() #reset simenv

#
# End
#
#################################################################


#################################################################
#
# 2. Compute v(s) using tabular TD(0)
#
#

#Compute v(s) using tabular TD(0) method
v=td0_v(cm,policy,0.9,1e-2,5000,100)

print('Value function (tabular):')
#plot v for all states, excluding absorbing terminal states
print(v[1:cm.numStates-1,0])
plt.plot(v[1:cm.numStates-1,0])
plt.show()

cm.reset() #reset simenv


#plot tabular vs func approx results for all states, excluding absorbing terminal states
#
plt.plot(v_fa,label='func approx')
plt.plot(v[1:cm.numStates-1,0],label='tabular')
plt.legend(loc="lower right")
plt.title('num_tiles={}'.format(num_tiles))
plt.show()

#
# End
#
#################################################################
