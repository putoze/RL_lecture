#
# Double Q-Learning with function approximation, for estimating the optimal policy & q-function
#
#  Implementation: Double q-learning, using epsilon-greedy approach
#

#
#"Skeleton" file for HW...
#
# Your task: Complete the missing code (see comments below in the code):
#            1. Implement semi-gradient double q-learning
#            2. Test your algorithm using the provided dbl_qlearn_fa_demo.py file
#


from random import Random
import torch

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
        
    def choose_action(self, q1_s, q2_s):
        '''
        Given q1_s=q1(state) q2_s=q2(state), make epsilon-greedy action choice
        '''
        #create epsilon-greedy policy (at current state only) from q1_s, q2_s
        policy=[self.rand_prob]*self.num_actions
        with torch.no_grad(): greedy_action=torch.argmax((q1_s+q2_s)/2)
        policy[greedy_action]=self.greedy_prob
        
        #choose random action based on e-greedy policy
        action=self.prng.choices(self.actions, weights=policy)[0]
        
        return action
    

#
#   Double Q-Learning with function approximation
#
def dbl_qlearn_fa(simenv, q1, q2, gamma, epsilon, alpha, num_episodes, max_episode_len, decayEpsilon=True, showPlots=False, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance
        q1, q2:  Models for double q's
        gamma :  Future discount factor, between 0 and 1
        epsilon: parameter for epsilon-greedy probabilities
        alpha :  Learning step size
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        decayEpsilon: If True, decay epsilon towards zero as reach num_episodes, else constant epsilon (default: True)
        showPlots:  If True, show plot of episode lengths during training, default: False
        prng_seed:  Seed for the random number generator        
    '''
    
    #initialize a few things
    #
    prng=Random()
    prng.seed(prng_seed)
    simenv.reset(seed=prng_seed)
                      
    actions=list(range(simenv.numActions)) #assume number of actions is same for all states
    
    #
    #You might need to add some additional initialization code here,
    #  depending on your specific implementation
    #

    egp=EpsilonGreedyPolicy(epsilon, actions, prng) #Epsilon-greedy policy helper
    
    
    ###########################
    #Start episode loop
    #
    episodeLengths=[]
    episodeRewards=[]

    for episode in range(num_episodes):
        if episode%100 == 0:
            print('Episode: {}'.format(episode))
        
        tot_reward=0
        
        #initial state & action (action according to policy)
        state=simenv.reset()
        
        #if we're decaying epsilon, compute now
        if decayEpsilon: egp.decay_epsilon(episode,num_episodes)
        
        #Run episode state-action-reward sequence to end
        #
        episode_length=0
        while episode_length < max_episode_len:

            #
            #Fill in the missing code here!  Semi-gradient double q-learning using function approximation
            #                                (you should use PyTorch capabilities in your implementation)
            # (Note: test your results with the dbl_qlearn_fa_demo.py file)
            #
            action = egp.choose_action(q1.forward(state), q2.forward(state))
            (next_state, reward, term_status, _) = simenv.step(action)

            q1_state = q1.forward(state)[action]
            q2_state = q2.forward(state)[action]

            q1_state.backward()
            q2_state.backward()

            prob = round(prng.random(), 1)
            with torch.no_grad():
                #with probability 0.5 => update q1
                if prob >= 0.5:
                    if term_status:
                        Ut = reward
                    else:
                        idx_max = torch.argmax(q1.forward(next_state))
                        Ut = reward + gamma*(q2.forward(next_state)[idx_max])
                    
                    for q1_weights in q1.weights:
                        q1_weights += alpha*(Ut - q1_state)*q1_weights.grad
                        q1_weights.grad.zero_()
                else:
                    if term_status:
                        Ut = reward
                    else:
                        idx_max = torch.argmax(q2.forward(next_state))
                        Ut = reward + gamma*(q1.forward(next_state)[idx_max])
                        
                    for q2_weights in q2.weights:
                        q2_weights += alpha*(Ut - q2_state)*q2_weights.grad
                        q2_weights.grad.zero_()
            tot_reward += reward
    
                            
            #check termination status from environment (reached terminal state?)
            if term_status: break  #if termination status is True, we've reached end of episode
            
            #move to next step in episode
            state=next_state
            episode_length+=1
        
        #update stats for later plotting
        episodeLengths.append(episode_length)
        episodeRewards.append(tot_reward)
        
    
    #if plot metrics was requested, do it now
    if showPlots:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot(episodeLengths)
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.subplot(212)
        plt.plot(episodeRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()
        #cleanup plots
        plt.cla()
        plt.close('all')
