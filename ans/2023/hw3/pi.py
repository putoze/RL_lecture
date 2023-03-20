#################################################################
#
# Compute optimal value and policy functions using policy iteration (PI) method
#
import numpy as np
from veval_matrix import veval_matrix

def pi(pssa, rsa, gamma, num_iters):
    #number of states & actions
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]

    #Initial policy is uniform random
    policy=np.zeros((nstates,nactions),dtype=np.float32)
    policy.fill(1.0/nactions)
    new_policy=None
    
    for i in range(num_iters):
        #First, evaluate the current policy
        # (note, using "exact" matrix solution rather than iterative
        #  approach here since our example cases are so small.)
        #
        v=veval_matrix(pssa,rsa,policy,gamma)
               
        #Next, update the policy using greedy lookahead (policy becomes deterministic here)
        # (Note: np.sum term below is matrix mult of each pssa action pane with v, with result put into (s, a) 2-d matrix form)
        #
        new_policy=np.argmax(rsa+gamma*np.sum(pssa*v,axis=1), axis=1)
        policy=np.zeros((nstates,nactions),dtype=np.float32)
        
        #update the policy matrix
        for j in range(nstates):
            policy[j,new_policy[j]]=1.0    
        
    #return optimal value and policy functions
    return (v,new_policy)