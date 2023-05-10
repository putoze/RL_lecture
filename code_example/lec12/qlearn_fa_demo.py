#
#Q-learning with function approximation
# - Demoed using OpenAi Acrobot & CartPole
#

from qlearn_fa import qlearn_fa, qlearn_fa_replay
import gym
import torch

show_plots=True


#Small feedforward neural network model for q (PyTorch)
# This NN architecture style is state in, q-per-action out
#
class MLP(torch.nn.Module):
    def __init__(self, numFeatures,numActions,useTanh=False):
        '''
        Parameters:
            numFeatures: Number of input features
            numActions: Number of output actions
        '''
        super().__init__()

        self.dense1=torch.nn.Linear(numFeatures,32)
        if useTanh: self.nonlin1=torch.nn.Tanh()
        else: self.nonlin1=torch.nn.ReLU()
        self.dense2=torch.nn.Linear(32,numActions)
 
    def forward(self,s):
        '''
        Compute value function q(s,a,w) by forward computation through MLP   
        '''
        feature_input=torch.tensor(s,dtype=torch.float32)

        #forward propagate input through network layers
        output=self.dense1(feature_input)
        output=self.nonlin1(output)
        output=self.dense2(output)

        return output

    @property
    def weights(self):
        '''
        Return model parameters
        '''
        return self.parameters()



#Environment parameter setup
# (uncomment out env param line below for environment you'd like to run)
#
#Cartpole
#env={'name': 'CartPole-v1', 'tanh': True, 'dir_base': './cartpole'}
#Acrobot
env={'name': 'Acrobot-v1', 'tanh': False, 'dir_base': './acrobot'}


#################################################################
#
# 1. Compute & demo optimal policy: Q-learning with function approximation
#    (generates animated .mp4 video to demo computed policy)
#
#
torch.manual_seed(456)  #let's make things repeatable! (only affects PyTorch neural-network param initialization in this demo)

simenv = gym.make(env['name'])
simenv.numActions=simenv.action_space.n
simenv.numFeatures=simenv.observation_space.shape[0]

q=MLP(simenv.numFeatures,simenv.numActions,useTanh=env['tanh'])

#Compute q(s,a) using Q-learning with function approximation
qlearn_fa(simenv,q,0.99,1.0,1e-3,3000,500,epsilon_decay_len=1500,decayEpsilon=True,showPlots=show_plots)

#run an episode using computed q(s,a)
from gym.wrappers import RecordVideo
simenv = RecordVideo(gym.make(env['name']), env['dir_base']+'_video')
state=simenv.reset(seed=789)

term_status=False
episode_len=0
while not term_status:
    action=int(torch.argmax(q.forward(state)))
    (next_state,reward,term_status,_)=simenv.step(action)

    if term_status: break #reached end of episode
    state=next_state
    episode_len+=1
print('Episode Length: {}'.format(episode_len))

simenv.close()
    
#
# End
#
#################################################################

#################################################################
#
# 2. Compute & demo optimal policy: Q-learning with function approximation & experience replay
#    (generates animated .mp4 video to demo computed policy)
#
#
torch.manual_seed(456)  #let's make things repeatable! (only affects PyTorch neural-network param initialization in this demo)

simenv = gym.make(env['name'])
simenv.numActions=simenv.action_space.n
simenv.numFeatures=simenv.observation_space.shape[0]

q=MLP(simenv.numFeatures,simenv.numActions,useTanh=env['tanh'])

#Compute q(s,a) using Q-learning with function approximation & experience replay   
qlearn_fa_replay(simenv,q,0.99,1.0,1e-3,3000,500,epsilon_decay_len=1500,decayEpsilon=True,showPlots=show_plots,
                     replay_size=5000, batch_size=10)

#run an episode using computed q(s,a)
from gym.wrappers import RecordVideo
simenv = RecordVideo(gym.make(env['name']), env['dir_base']+'_video_replay')
state=simenv.reset(seed=789)

term_status=False
episode_len=0
while not term_status:
    action=int(torch.argmax(q.forward(state)))
    (next_state,reward,term_status,_)=simenv.step(action)

    if term_status: break #reached end of episode
    state=next_state
    episode_len+=1
print('Episode Length: {}'.format(episode_len))

simenv.close()
    
#
# End
#
#################################################################
