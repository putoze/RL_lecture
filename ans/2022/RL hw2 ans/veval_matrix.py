#################################################################
#
#     Compute the value function for given policy solving Bellman
#     eqn as system of linear equations (linear algebra approach)
#

import numpy as np

def veval_matrix(pssa, rsa, policy, gamma):
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
    
    
    #solve for value function using system of linear eqns: v=(I-Ppi)^-1*Rpi
    #
    ident=np.identity(nstates, dtype=np.float32)
    v=np.linalg.inv(ident-gamma*Ppi)@Rpi
    
    #return computed value function
    return v