#
# SARSA control, for estimating the optimal policy & q-function
#   q(s,a) [matrix: numStates x numActions], policy, a=pi(s) [column vector: numStates x 1]
#
#  Implementation: SARSA w/ epsilon-greedy
#

from random import Random
import numpy as np


class EpsilonGreedyPolicy:
    '''
    Helper class to create/manage/use epsilon-greedy policies with q
    '''
    def __init__(self, epsilon, actions, prng):
        self.epsilon0=epsilon
        self.epsilon=epsilon
        self.actions=list(actions) #assume number of actions same for all states
        self.num_actions=len(actions)
        self.prng=prng
        
        #pre-compute a few things for efficiency
        self.greedy_prob=1.0-epsilon+epsilon/self.num_actions
        self.rand_prob=epsilon/self.num_actions
        
    def decay_epsilon(self, episode, num_episodes):
        self.epsilon=self.epsilon0*(num_episodes - episode)/num_episodes
        self.greedy_prob=1.0-self.epsilon+self.epsilon/self.num_actions
        self.rand_prob=self.epsilon/self.num_actions
        
    def choose_action(self, q, state):
        '''
        Given q & state, make epsilon-greedy action choice
        '''
        #create epsilon-greedy policy (at current state only) from q
        policy=[self.rand_prob]*self.num_actions
        greedy_action=np.argmax(q[state])
        policy[greedy_action]=self.greedy_prob
        
        #choose random action based on e-greedy policy
        action=self.prng.choices(self.actions, weights=policy)[0]
        
        return action
    

#
#   SARSA control
#
def sarsa(simenv, init_state, gamma, epsilon, alpha, num_episodes, max_episode_len, decayEpsilon=True, showPlots=False, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance (e.g., Cat_and_Mouse)
        init_state: Initial state for episodes (enumerated, int)
        gamma :  Future discount factor, between 0 and 1
        epsilon: parameter for epsilon-greedy probabilities
        alpha :  Learning step size
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        decayEpsilon: If True, decay epsilon towards zero as reach num_episodes, else constant epsilon (default: True)
        showPlots:  If True, show plot of episode lengths during training, default: False
        prng_seed:  Seed for the random number generator
        
    Return value:
        The final value of pi(s) & q(s,a), the estimated optimal policy & q functions
    '''
    #initialize a few things
    #
    prng=Random()
    prng.seed(prng_seed)
                 
    np.random.seed(prng_seed)
    q=np.zeros((simenv.numStates,simenv.numActions),dtype=np.float32)
                  
    actions=list(range(simenv.numActions)) #assume number of actions is same for all states
    
    #Epsilon-greedy policy helper
    egp=EpsilonGreedyPolicy(epsilon, actions, prng)
    
    #Start episode loop
    #
    episodeLengths=[]
    for episode in range(num_episodes):
        #initial state & action (action according to policy)
        #
        state=init_state
        simenv.initState(state)
        action=egp.choose_action(q,state)
        
        #if we're decaying epsilon, compute now
        if decayEpsilon: egp.decay_epsilon(episode,num_episodes)
        
        #Run episode state-action-reward sequence to end
        #
        episode_length=0
        while episode_length < max_episode_len:            
            #take action, get reward, next state, termination status
            (next_state,reward,term_status)=simenv.step(action)
            
            #choose next action (in next_state) from e-greedy policy
            next_action=egp.choose_action(q,next_state)
            
            #update q every time step (SARSA update)
            q[state,action]=q[state,action]+alpha*(reward+gamma*q[next_state,next_action]-q[state,action]) #incremental update of q(s,a)

            
            #check termination status from environment (reached terminal state?)
            if term_status: break  #if termination status is True, we've reached end of episode
            
            #move to next step in episode
            state=next_state
            action=next_action
            
            episode_length+=1

        #update stats for later plotting    
        episodeLengths.append(episode_length)


    #if plot metrics was requested, do it now
    if showPlots:
        import matplotlib.pyplot as plt
        plt.plot(episodeLengths)
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.show()
        #cleanup plots
        plt.cla()
        plt.close('all')
            
                
    #return final greedy deterministic policy (1-D column vec)
    policy=np.argmax(q,axis=1)
    policy.shape=(simenv.numStates,1)
    
    #  Note: For non-decaying epsilon cases, i.e. constant epsilon, it is debatable
    #  whether it is technically correct to "harden" the final computed epison-greedy policies,
    #  however, let's assume that for our constant-epsilon demo cases here, we will use small epsilon values
    #  and the final computed policies will be adequately represented by greedy deterministic results
   
    return (policy, q)

