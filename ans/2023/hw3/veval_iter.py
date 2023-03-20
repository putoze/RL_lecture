#################################################################
#
#     Compute the value function for given policy solving Bellman
#     eqn iteratively (synchronous & asynchronous)
#

import numpy as np

#Synchronous approach -- i.e., update all states simultaneously each iteration
#
def veval_iter_sync(pssa, rsa, policy, gamma, num_iters):
    #number of states & actions
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]
        
    #First, let's compute a few useful intermediate matrices,
    # r(s,a) and p(s',s,a) averaged over policy actions    
    Rpi=np.sum(policy*rsa,axis=1)
    Rpi.shape=(nstates,1) #reshape into column vector
    
    Ppi=np.zeros((nstates,nstates),dtype=np.float32)
    for j in range(nstates):
        Ppi[j]=np.sum(policy[j]*pssa[j],axis=1)
    
    
    #compute value function iteratively (synchronous)
    v=np.zeros((nstates,1),np.float32)
    for i in range(num_iters):
        v_new=Rpi+gamma*Ppi@v
        v=v_new

    #return computed value function    
    return v


#Asynchronous approach -- i.e., update all states one-by-one sequentially each iteration
#
def veval_iter_async(pssa, rsa, policy, gamma, num_iters):
    #number of states & actions
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]
        
    #First, let's compute a few useful intermediate matrices,
    # r(s,a) and p(s',s,a) averaged over policy actions    
    Rpi=np.sum(policy*rsa,axis=1)
    Rpi.shape=(nstates,1) #reshape into column vector
    
    Ppi=np.zeros((nstates,nstates),dtype=np.float32)
    for j in range(nstates):
        Ppi[j]=np.sum(policy[j]*pssa[j],axis=1)
    
    
    #compute value function iteratively (asynchronous)
    v=np.zeros((nstates,1),np.float32)
    for i in range(num_iters):
        for s in range(nstates):
            v[s]=Rpi[s]+gamma*Ppi[s,:]@v

    #return computed value function    
    return v
