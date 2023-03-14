#
#Based on Gridworld Example 3.5c from Sutton & Barto 2nd Ed.
#
# We first build a DP model of the Gridworld, then solve for the following:
#  (1) Compute value function by solving Bellman eqn as system of linear equations (linear algebra approach)
#  (2) Compute value function using iterative evaluation (synchronous & asynchronous)
#  (3) Compute optimal value and policy functions using policy iteration (PI) method
#  (4) Compute optimal value and policy functions using value iteration (VI) method
#
# The Gridworld is a 2-D 5x5 grid with 25 states labeled as follows:
#   0  1  2  3  4
#   5  6  7  8  9
#   10 11 12 13 14
#   15 16 17 18 19
#   20 21 22 23 24
#
# In each cell 4 actions are possible, North, South, West, East. Transitions and rewards
#  are deterministic.  If a move goes off the edge of the map, agent gets a reward of -1
#  and original state is left unchanged. All other actions result in a reward of 0, with
#  the exception of two special states A (state 1) and B (state 3).  From state A all
#  actions result in a reward of 10 and take the agent to state A' (state 21).  From state
#  B all actions yield a reward of 5 and take the agent to B' (state 13).
#  (for further details, please see Sutton & Barto example 3.5c)

import numpy as np
from enum import IntEnum

#numpy print options
np.set_printoptions(precision=2)


#################################################################
#
# Create the Gridworld transition, action & reward model
#

#number of states/actions for this problem
nstates=25
nactions=4

#future discount rate
gamma=0.9

#Action mapping
#    
class Action(IntEnum):
    North=0
    South=1
    West=2
    East=3
A=Action #alias for shorter names!


#The reward vector r(s,a)
#
rsa=np.zeros((nstates,nactions),dtype=np.float32)
for i in range(5):
    rsa[i,A.North]=-1.0
for i in range(20,25):
    rsa[i,A.South]=-1.0
for i in range(0,25,5):
    rsa[i,A.West]=-1.0
for i in range(4,25,5):
    rsa[i,A.East]=-1.0
#special transition A->A' (state 1->21)
for i in range(nactions): rsa[1,i]=10.0
#special transition B->B' (state 3->13)
for i in range(nactions): rsa[3,i]=5.0


#state-action transition table p(s',s,a)
#
pssa=np.zeros((nstates,nstates,nactions),dtype=np.float32)

#move-north pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(5):
        pssa[i,i,A.North]=1.0
    else:
        pssa[i,i-5,A.North]=1.0
        
#move-south pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(20,25):
        pssa[i,i,A.South]=1.0
    else:
        pssa[i,i+5,A.South]=1.0
        
#move-west pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(0,25,5):
        pssa[i,i,A.West]=1.0
    else:
        pssa[i,i-1,A.West]=1.0
        
#move-east pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(4,25,5):
        pssa[i,i,A.East]=1.0
    else:
        pssa[i,i+1,A.East]=1.0
        
#special A, B cells
for move in range(nactions):
    pssa[1,21,move]=1.0
    pssa[3,13,move]=1.0

#
# End Gridworld model creation
#
#################################################################

#################################################################
#
# 1. Compute the value function for random policy solving Bellman
#     eqn as system of linear equations (linear algebra approach)
#
try:
    from veval_matrix import veval_matrix
    
    #Policy function for uniform random policy
    policy=np.zeros((nstates,nactions),dtype=np.float32)
    policy.fill(1.0/nactions) #4 directions, 25% probability each direction

    #solve for value function
    v=veval_matrix(pssa,rsa,policy,gamma)
    
    print('Value function computed as system of linear equations:')
    v.shape=(5,5) #reshape value vector to match 2-D gridworld shape
    print(v)
except ImportError:
    #module not found, skipping
    pass

#
# End value function computation for random policy
#
#################################################################

#################################################################
#
# 2.  Compute the value function for random policy solving Bellman
#     eqn iteratively (synchronous & asynchronous)
#
try:
    from veval_iter import veval_iter_sync, veval_iter_async

    #Policy function for uniform random policy
    policy=np.zeros((nstates,nactions),dtype=np.float32)
    policy.fill(1.0/nactions) #4 directions, 25% probability each direction

    #solve for the value function (synchronous iter)
    v=veval_iter_sync(pssa,rsa,policy,gamma,num_iters=15)
    
    print('\nValue function computed by iterative policy evaluation (synchronous):')
    v.shape=(5,5) #reshape value vector to match 2-D gridworld shape
    print(v)
    
    #solve for the value function (asynchronous iter)
    v=veval_iter_async(pssa,rsa,policy,gamma,num_iters=25)
    
    print('\nValue function solved by iterative policy evaluation (asynchronous):')
    v.shape=(5,5) #reshape value vector to match 2-D gridworld shape
    print(v)  
except ImportError:
    #module not found, skipping
    pass

#
# End value function computation for random policy
#
#################################################################

#################################################################
#
# 3. Compute optimal value and policy functions using policy iteration (PI) method
#
try:
    from pi import pi
    
    #solve for optimal value & policy functions
    (v,policy)=pi(pssa,rsa,gamma,num_iters=5)
    
    print('\nOptimal value function (PI method):')
    v.shape=(5,5)
    print(v)

    print('\nOptimal policy (PI method):')
    policy.shape=(5,5)
    print(policy)
except ImportError:
    #module not found, skipping
    pass
    
#
# End optimal value & policy computation
#
#################################################################

#################################################################
#
# 4. Compute optimal value and policy functions using value iteration (VI) method
#
try:
    from vi import vi
    
    #solve for optimal value & policy functions
    (v,policy)=vi(pssa,rsa,gamma,num_iters=40)
    
    print('\nOptimal value function (VI method):')
    v.shape=(5,5)
    print(v)

    print('\nOptimal policy (VI method):')
    policy.shape=(5,5)
    print(policy)
    
except ImportError:
    #module not found, skipping
    pass  
    
#
# End optimal value & policy computation
#
#################################################################
