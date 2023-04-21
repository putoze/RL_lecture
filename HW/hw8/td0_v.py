#
# TD(0) for estimating the state-value function,
#   v(s) [column vector: numStates x 1], given a policy, pi(a|s) [matrix: numStates x numActions]
#

from random import Random
import numpy as np

def td0_v(simenv, policy, gamma, alpha, num_episodes, max_episode_len, exploring_starts=False, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance (e.g., Cat_and_Mouse)
        policy:  Policy action-probability matrix, numStates x numActions
        gamma :  Future discount factor, between 0 and 1
        alpha :  Learning step size
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        exploring_starts:  False: Use init state provided by simenv, True: Use random init states for episodes
        prng_seed:  Seed for the random number generator
        
    Return value:
        The final value of v(s), the value function
    '''
    #initialize a few things
    #
    prng=Random()
    prng.seed(prng_seed)
                 
    v=np.zeros((simenv.numStates,1),dtype=np.float32)  #the value function
    actions=list(range(simenv.numActions)) #assume number of actions is same for all states
    
    #Start episode loop
    #
    for episode in range(num_episodes):
        #if using exploring starts, choose random init state,
        #  else use simenv-provided init state
        state=None
        if exploring_starts:
            state=prng.randint(0,simenv.numStates-1)
            simenv.initState(state)
        else:
            simenv.initState()
            state=simenv.currentState()
        
        #Run episode sequence to end
        #
        episode_length=0
        while episode_length < max_episode_len:
            #sample the policy actions at current state
            action=prng.choices(actions, weights=policy[state])[0]
            
            #take action, get reward, next state, termination status
            (next_state,reward,term_status)=simenv.step(action)
            
            #update v(s) estimate every time step
            v[state]=v[state]+alpha*(reward+gamma*v[next_state]-v[state])
            
            #check termination status from environment (reached terminal state?)
            if term_status: break  #if termination status is True, we've reached end of episode
            
            state=next_state
            episode_length+=1
        
    return v


