#
#Policy-gradient with Monte Carlo (REINFORCE)
# - Demoed using OpenAI CartPole & Acrobot
#

from pg_mc import pg_mc
import gym
import torch
from torch.distributions import Categorical
torch.manual_seed(456)  #let's make things repeatable!
show_plots=True


#Small feedforward neural network model for policy (PyTorch)
# This NN architecture style is state in, probability-per-action out (using softmax)
#
class MLP(torch.nn.Module):
    def __init__(self, numFeatures,numActions):
        '''
        Parameters:
            numFeatures: Number of input features
            numActions: Number of output actions
        '''
        super().__init__()

        self.dense1=torch.nn.Linear(numFeatures,32)
        self.relu1=torch.nn.ReLU()
        self.dense2=torch.nn.Linear(32,numActions)
        self.softmax1=torch.nn.Softmax(dim=0)
 
    def forward(self,s):
        '''
        Compute policy function pi(a|s,w) by forward computation through MLP   
        '''
        feature_input=torch.tensor(s,dtype=torch.float32)

        #forward propagate input through network layers
        output=self.dense1(feature_input)
        output=self.relu1(output)
        output=self.dense2(output)
        output=self.softmax1(output)

        return output

    def choose_action(self,s,returnLogpi=True):
        '''
        Sample an action at state s, using current policy
        
        Returns chosen action, and optionally the computed PG log pi term
        '''
        pi_s = self.forward(s)
        prob_model = Categorical(pi_s)
        action = prob_model.sample()   #sample an action from current policy probabilities
        
        if not returnLogpi:
            return action.item()
        else:
            log_pi=torch.log(pi_s[action]) #log pi
            return (action.item(), log_pi)

    

#################################################################
#
# 1. Compute & demo optimal policy for OpenAI CartPole environment
#    (generates animated .mp4 video to demo computed policy)
#
#

simenv = gym.make('CartPole-v1')
simenv.numActions=simenv.action_space.n
simenv.numFeatures=simenv.observation_space.shape[0]

policy=MLP(simenv.numFeatures,simenv.numActions)
    
#Compute policy using Policy Gradient with Monte Carlo (REINFORCE)
pg_mc(simenv,policy,0.99,1e-3,2000,500,showPlots=show_plots)


#run an episode using computed policy
from gym.wrappers import RecordVideo
simenv = RecordVideo(gym.make('CartPole-v1'), './cartpole_video')
state=simenv.reset(seed=789)

term_status=False
episode_len=0
while not term_status:
    action=int(torch.argmax(policy.forward(state)))  #convert to "greedy" deterministic policy
    (next_state,reward,term_status,_)=simenv.step(action)
    
    if term_status: break #reached end of episode
    state=next_state
    episode_len+=1
print('Episode Length: {}\n'.format(episode_len))

simenv.close()
    
#
# End
#
#################################################################


#################################################################
#
# 2. Compute & demo optimal policy for OpenAI Acrobot environment
#    (generates animated .mp4 video to demo computed policy)
#
#

simenv = gym.make('Acrobot-v1')
simenv.numActions=simenv.action_space.n
simenv.numFeatures=simenv.observation_space.shape[0]

policy=MLP(simenv.numFeatures,simenv.numActions)
    
#Compute policy using Policy Gradient with Monte Carlo (REINFORCE)
pg_mc(simenv,policy,0.99,1e-3,2000,500,showPlots=show_plots)


#run an episode using computed policy
from gym.wrappers import RecordVideo
simenv = RecordVideo(gym.make('Acrobot-v1'), './acrobot_video')
state=simenv.reset(seed=789)

term_status=False
episode_len=0
while not term_status:
    action=int(torch.argmax(policy.forward(state)))  #convert to "greedy" deterministic policy
    (next_state,reward,term_status,_)=simenv.step(action)
    
    if term_status: break #reached end of episode
    state=next_state
    episode_len+=1
print('Episode Length: {}\n'.format(episode_len))

simenv.close()
    
#
# End
#
#################################################################

