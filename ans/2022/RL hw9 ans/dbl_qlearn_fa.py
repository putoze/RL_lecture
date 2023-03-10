#
# Double Q-Learning with function approximation, for estimating the optimal policy & q-function
#
#  Implementation: Double q-learning, using epsilon-greedy approach
#

from random import Random
from collections import deque
import torch

class EpsilonGreedyPolicy:
    '''
    Helper class to create/manage/use epsilon-greedy policies with q
    '''
    def __init__(self, epsilon, epsilon_decay_len, actions, prng):
        self.epsilon0=epsilon
        self.epsilon=epsilon
        self.epsilon_decay_len=epsilon_decay_len
        self.actions=list(actions) #assume number of actions same for all states
        self.num_actions=len(actions)
        self.prng=prng
        
        #pre-compute a few things for efficiency
        self.greedy_prob=1.0-epsilon+epsilon/self.num_actions
        self.rand_prob=epsilon/self.num_actions
        
    def decay_epsilon(self, episode):
        self.epsilon=self.epsilon0*(self.epsilon_decay_len - episode)/self.epsilon_decay_len
        if self.epsilon < 0: self.epsilon=0
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
def dbl_qlearn_fa(simenv, q1, q2, gamma, epsilon, alpha, num_episodes, max_episode_len, 
                  window_len=100, term_thresh=None, epsilon_decay_len=None, decayEpsilon=True, showPlots=False, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance
        q1, q2:  Models for double q's
        gamma :  Future discount factor, between 0 and 1
        epsilon: parameter for epsilon-greedy probabilities
        alpha :  Learning step size
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        window_len: Window size for total rewards windowed average (averaged over multiple episodes)
        term_thresh: If windowed average > term_thresh, stop (if None, run until num_episodes)
        epsilon_decay_len: End point (episode count) for epsilon decay (epsilon=0 after endpoint. If None, endpoint=num_episodes)
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

    mse=torch.nn.MSELoss() #mse cost function for J(w)
    optim1=torch.optim.Adam(q1.parameters(),lr=alpha) #optimizers for q1, q2
    optim2=torch.optim.Adam(q2.parameters(),lr=alpha)

    if epsilon_decay_len is None: epsilon_decay_len=num_episodes
    egp=EpsilonGreedyPolicy(epsilon, epsilon_decay_len, actions, prng) #Epsilon-greedy policy helper


    ###########################
    #Start episode loop
    #
    episodeLengths=[]
    episodeRewards=[]
    averagedRewards=[]

    for episode in range(num_episodes):
        if episode%100 == 0:
            print('Episode: {}'.format(episode))

        tot_reward=0

        #initial state & action (action according to policy)
        state=simenv.reset()

        #if we're decaying epsilon, compute now
        if decayEpsilon: egp.decay_epsilon(episode)

        #Run episode state-action-reward sequence to end
        #
        episode_length=0
        while episode_length < max_episode_len:
            #randomly choose which q (q1, q2) will be updated this step
            # (qup is the model that gets updated & qfx is the one that is fixed for this step)
            if prng.random() < 0.5:
                qup=q1
                qfx=q2
                optim=optim1
            else:
                qup=q2
                qfx=q1
                optim=optim2

            #evaluate q(state)
            qup_s=qup.forward(state)
            qfx_s=qfx.forward(state)

            #choose action, based on e-greedy policy
            action=egp.choose_action(qup_s,qfx_s)

            #take action, get reward, next state, termination status
            (next_state,reward,term_status,_)=simenv.step(action)
            tot_reward+=reward

            #Compute q(state,action)
            qup_sa=qup_s[action]

            #compute target
            with torch.no_grad():
                if term_status:
                    #if next_state is terminal state, assume q(next_state)=0 
                    target=torch.tensor(reward)
                else:
                    q_s1_max=qfx.forward(next_state)[torch.argmax(qup.forward(next_state))] #don't include q_s1 in gradient, this is a semi-gradient method!
                    target=reward+gamma*q_s1_max

            #compute cost & gradients, update model weights
            jw=mse(target,qup_sa)  #J(w) (for 1 sample)
            jw.backward()          #compute gradients
            optim.step()           #step the optimizer, update weights
            optim.zero_grad()      #zero out gradients (otherwise PyTorch will keep accumulating gradients)


            #check termination status from environment (reached terminal state?)
            if term_status: break  #if termination status is True, we've reached end of episode

            #move to next step in episode
            state=next_state
            episode_length+=1

        #update stats for later plotting
        episodeLengths.append(episode_length)
        episodeRewards.append(tot_reward)
        avg_tot_reward=sum(episodeRewards[-window_len:])/window_len
        averagedRewards.append(avg_tot_reward)

        if episode%100 == 0:
            print('\tAvg reward: {}'.format(avg_tot_reward))

        #if termination condition was specified, check it now
        if (term_thresh != None) and (avg_tot_reward >= term_thresh): break


    #if plot metrics was requested, do it now
    if showPlots:
        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.plot(episodeLengths)
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.subplot(312)
        plt.plot(episodeRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.subplot(313)
        plt.plot(averagedRewards)
        plt.xlabel('Episode')
        plt.ylabel('Avg Total Reward')
        plt.show()
        #cleanup plots
        plt.cla()
        plt.close('all')



#
#  Alternate version that uses experience replay w/ mini-batches
#
#
#   Double Q-Learning with function approximation
#
# def dbl_qlearn_fa(simenv, q1, q2, gamma, epsilon, alpha, num_episodes, max_episode_len, 
#                   window_len=100, term_thresh=None, epsilon_decay_len=None,
#                   replay_size=5000, batch_size=20, decayEpsilon=True, showPlots=False, prng_seed=789):
#     '''
#     Parameters:
#         simenv:  Simulation environment instance
#         q1, q2:  Models for double q's
#         gamma :  Future discount factor, between 0 and 1
#         epsilon: parameter for epsilon-greedy probabilities
#         alpha :  Learning step size
#         num_episodes: Number of episodes to run
#         max_episode_len: Maximum allowed length of a single episode
#         window_len: Window size for total rewards windowed average (averaged over multiple episodes)
#         term_thresh: If windowed average > term_thresh, stop (if None, run until num_episodes)
#         epsilon_decay_len: End point (episode count) for epsilon decay (epsilon=0 after endpoint. If None, endpoint=num_episodes)
#         replay_size: Size of the experience replay buffer
#         batch_size: Size of the mini-batches sampled from the replay buffer
#         decayEpsilon: If True, decay epsilon towards zero as reach num_episodes, else constant epsilon (default: True)
#         showPlots:  If True, show plot of episode lengths during training, default: False
#         prng_seed:  Seed for the random number generator        
#     '''
#
#     #initialize a few things
#     #
#     prng=Random()
#     prng.seed(prng_seed)
#     simenv.reset(seed=prng_seed)
#
#     actions=list(range(simenv.numActions)) #assume number of actions is same for all states
#
#     mse=torch.nn.MSELoss() #mse cost function for J(w)
#     optim1=torch.optim.Adam(q1.parameters(),lr=alpha) #optimizers for q1, q2
#     optim2=torch.optim.Adam(q2.parameters(),lr=alpha)
#
#     if epsilon_decay_len is None: epsilon_decay_len=num_episodes
#     egp=EpsilonGreedyPolicy(epsilon, epsilon_decay_len, actions, prng) #Epsilon-greedy policy helper
#
#     replayBuffer=deque(maxlen=replay_size) #experience replay buffer
#
#
#     ###########################
#     #Start episode loop
#     #
#     episodeLengths=[]
#     episodeRewards=[]
#     averagedRewards=[]
#
#     for episode in range(num_episodes):
#         if episode%100 == 0:
#             print('Episode: {}'.format(episode))
#
#         tot_reward=0
#
#         #initial state & action (action according to policy)
#         state=simenv.reset()
#
#         #if we're decaying epsilon, compute now
#         if decayEpsilon: egp.decay_epsilon(episode)
#
#         #Run episode state-action-reward sequence to end
#         #
#         episode_length=0
#         while episode_length < max_episode_len:
#             #randomly choose which q (q1, q2) will be updated this step
#             # (qup is the model that gets updated & qfx is the one that is fixed for this step)
#             if prng.random() < 0.5:
#                 qup=q1
#                 qfx=q2
#                 optim=optim1
#             else:
#                 qup=q2
#                 qfx=q1
#                 optim=optim2
#
#             #evaluate q(state) for choosing action
#             with torch.no_grad():
#                 qup_s=qup.forward(state)
#                 qfx_s=qfx.forward(state)
#
#             #choose action, based on e-greedy policy
#             action=egp.choose_action(qup_s,qfx_s)
#
#             #take action, get reward, next state, termination status
#             (next_state,reward,term_status,_)=simenv.step(action)
#             tot_reward+=reward
#
#             #insert current sample into replay buffer
#             replayBuffer.append((state,action,reward,next_state,term_status))
#
#             #sample mini-batch from replayBuffer
#             # if buffer size < mini-batch size, then sample entire buffer
#             sample_size=len(replayBuffer)
#             if sample_size >= batch_size: sample_size=batch_size
#             replay=prng.sample(replayBuffer,sample_size)
#
#             targets=[]
#             qup_sas=[]
#
#             for e in replay:
#                 (estate,eaction,ereward,enext_state,eterm_status)=e
#
#                 #Compute q(state,action) (w/ gradient tracking)
#                 qup_s=qup.forward(estate)
#                 qup_sa=qup_s[eaction]
#                 qup_sas.append(qup_sa)
#
#                 #compute targets
#                 with torch.no_grad():
#                     if eterm_status:
#                         #if next_state is terminal state, assume q(next_state)=0 
#                         target=torch.tensor(ereward)
#                     else:
#                         q_s1_max=qfx.forward(enext_state)[torch.argmax(qup.forward(enext_state))] #don't include q_s1 in gradient, this is a semi-gradient method!
#                         target=ereward+gamma*q_s1_max
#                     targets.append(target)
#
#             #stack batch of targets & q's into tensors for mse & optimization steps
#             tstack=torch.stack(targets)
#             qstack=torch.stack(qup_sas)
#
#             #compute cost & gradients, update model weights
#             jw=mse(tstack,qstack)  #J(w) (for mini-batch)
#             jw.backward()          #compute gradients
#             optim.step()           #step the optimizer, update weights
#             optim.zero_grad()      #zero out gradients (otherwise PyTorch will keep accumulating gradients)
#
#
#             #check termination status from environment (reached terminal state?)
#             if term_status: break  #if termination status is True, we've reached end of episode
#
#             #move to next step in episode
#             state=next_state
#             episode_length+=1
#
#         #update stats for later plotting
#         episodeLengths.append(episode_length)
#         episodeRewards.append(tot_reward)
#         avg_tot_reward=sum(episodeRewards[-window_len:])/window_len
#         averagedRewards.append(avg_tot_reward)
#
#         if episode%100 == 0:
#             print('\tAvg reward: {}'.format(avg_tot_reward))
#
#         #if termination condition was specified, check it now
#         if (term_thresh != None) and (avg_tot_reward >= term_thresh): break
#
#
#     #if plot metrics was requested, do it now
#     if showPlots:
#         import matplotlib.pyplot as plt
#         plt.subplot(311)
#         plt.plot(episodeLengths)
#         plt.xlabel('Episode')
#         plt.ylabel('Length')
#         plt.subplot(312)
#         plt.plot(episodeRewards)
#         plt.xlabel('Episode')
#         plt.ylabel('Total Reward')
#         plt.subplot(313)
#         plt.plot(averagedRewards)
#         plt.xlabel('Episode')
#         plt.ylabel('Avg Total Reward')
#         plt.show()
#         #cleanup plots
#         plt.cla()
#         plt.close('all')
