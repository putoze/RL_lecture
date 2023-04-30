#
#Double Q-learning with function approximation
# - Demoed using OpenAi CartPole
#

from dbl_qlearn_fa import dbl_qlearn_fa
import gym
import torch
torch.manual_seed(456)  #let's make things repeatable! (only affects PyTorch neural-network param initialization in this demo)
show_plots=True


#Small feedforward neural network model for q (PyTorch)
# This NN architecture style is state in, q-per-action out
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
 
    def forward(self,s):
        '''
        Compute value function q(s,a,w) by forward computation through MLP   
        '''
        feature_input=torch.tensor(s,dtype=torch.float32)

        #forward propagate input through network layers
        output=self.dense1(feature_input)
        output=self.relu1(output)
        output=self.dense2(output)

        return output

    @property
    def weights(self):
        '''
        Return model parameters
        '''
        return self.parameters()


#################################################################
#
# 1. Compute & demo optimal policy for OpenAI CartPole environment
#    (generates animated .mp4 video to demo computed policy)
#
#

simenv = gym.make('CartPole-v1')
simenv.numActions=simenv.action_space.n
simenv.numFeatures=simenv.observation_space.shape[0]

q1=MLP(simenv.numFeatures,simenv.numActions)
q2=MLP(simenv.numFeatures,simenv.numActions)
    
#Compute q(s,a) using Double Q-learning with function approximation
dbl_qlearn_fa(simenv,q1,q2,0.99,1.0,1e-3,5000,500,decayEpsilon=True,showPlots=show_plots)


#run an episode using computed q(s,a)
from gym.wrappers import RecordVideo
simenv = RecordVideo(gym.make('CartPole-v1'), './cartpole_video')
state=simenv.reset(seed=789)

term_status=False
episode_len=0
while not term_status:
    action=int(torch.argmax((q1.forward(state)+q2.forward(state))/2))
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

