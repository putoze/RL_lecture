#################################################################
#
# Compute optimal value and policy functions using value iteration (VI) method
#
import numpy as np

def vi(pssa, rsa, gamma, num_iters):
    #number of states & actions
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]

    #compute optimal value function using value iteration (VI) method (synchronous)
    # (Note: np.sum term below is matrix mult of each pssa action pane with v, with result put into (s, a) 2-d matrix form)
    #   
    v=np.zeros((nstates,1),np.float32)
    for i in range(num_iters):
        v_new=np.max(rsa+gamma*np.sum(pssa*v,axis=1), axis=1)
        v_new.shape=(nstates,1)
        v=v_new
               
    #Compute the optimal policy using greedy lookahead
    #
    policy=np.argmax(rsa+gamma*np.sum(pssa*v,axis=1), axis=1) 
        
    #return optimal value and policy functions
    return (v,policy)