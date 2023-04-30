#
# TD(0) for estimating the state-value function given a policy pi(a|s)
#   Uses parametric function approximation for v -> v(s,w), incremental semi-gradient approach for learning weights
#

from random import Random
import numpy as np
import torch

def td0_v_fa(simenv, vmodel, policy, gamma, alpha, num_episodes, max_episode_len, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance (e.g., Cat_and_Mouse)
        vmodel:  Model instance for v(s,w) parametric function approximation.
        policy:  Policy action-probability matrix, numStates x numActions
        gamma :  Future discount factor, between 0 and 1
        alpha :  Learning step size (used in gradient-descent step)
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        prng_seed:  Seed for the random number generator       
     '''
    #initialize a few things
    #
    prng=Random()
    prng.seed(prng_seed)

    actions=list(range(simenv.numActions)) #assume number of actions is same for all states

    #Start episode loop
    #
    for episode in range(num_episodes):
        #use simenv-provided init state
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

            #compute v(s) & v(s) gradients
            v_s=vmodel.v(state)     #v(s,w)
            v_s.backward()          #compute gradients for v(s,w)

            with torch.no_grad(): #don't track gradients during update step!
                v_s1=vmodel.v(next_state)  #don't include v_s1 in gradient, this is a semi-gradient method!
                #iterate through all parameter groups (e.g., PyTorch nn layer params)
                for weights in vmodel.parameters():
                    if term_status:
                        #if next_state is terminal state, assume v(next_state)=0 
                        weights+=alpha*(reward-v_s)*weights.grad
                    else:
                        weights+=alpha*(reward+gamma*v_s1-v_s)*weights.grad
                    weights.grad.zero_() #zero out gradients (otherwise PyTorch will keep accumulating gradients)

            #check termination status from environment (reached terminal state?)
            if term_status: break  #if termination status is True, we've reached end of episode

            state=next_state
            episode_length+=1

        if episode%500 == 0:
            print('\tEpisode: {}'.format(episode))


    return


#
# If you prefer using the PyTorch built-in cost functions & optimizers,
#   here's an alternate implementation
#
#
# def td0_v_fa(simenv, vmodel, policy, gamma, alpha, num_episodes, max_episode_len, prng_seed=789):
#     '''
#     Parameters:
#         simenv:  Simulation environment instance (e.g., Cat_and_Mouse)
#         vmodel:  Model instance for v(s,w) parametric function approximation.
#                    Note: Expect a class that has a weights attribute containing
#                    a list of all trainable parameter groups (like PyTorch nn classes)
#                    and a v(s) method that computes v(s,w) given a state s
#         policy:  Policy action-probability matrix, numStates x numActions
#         gamma :  Future discount factor, between 0 and 1
#         alpha :  Learning step size (used in gradient-descent step)
#         num_episodes: Number of episodes to run
#         max_episode_len: Maximum allowed length of a single episode
#         prng_seed:  Seed for the random number generator       
#      '''
#     #initialize a few things
#     #
#     prng=Random()
#     prng.seed(prng_seed)
#
#     actions=list(range(simenv.numActions)) #assume number of actions is same for all states
#
#     mse=torch.nn.MSELoss() #mse cost function for J(w)
#     optim=torch.optim.SGD(vmodel.parameters(),lr=alpha) #optimizer
#
#     #Start episode loop
#     #
#     for episode in range(num_episodes):
#         #use simenv-provided init state
#         simenv.initState()
#         state=simenv.currentState()
#
#         #Run episode sequence to end
#         #
#         episode_length=0
#         while episode_length < max_episode_len:
#             #sample the policy actions at current state
#             action=prng.choices(actions, weights=policy[state])[0]
#
#             #take action, get reward, next state, termination status
#             (next_state,reward,term_status)=simenv.step(action)
#
#             #compute v(s)
#             v_s=vmodel.v(state)     #v(s,w)
#
#             with torch.no_grad(): #don't track gradients
#                 if term_status:
#                     #if next_state is terminal state, assume v(next_state)=0
#                     target=torch.tensor([reward])
#                 else:
#                     v_s1=vmodel.v(next_state)  #don't include v_s1 in gradient, this is a semi-gradient method!
#                     target=reward+gamma*v_s1
#
#             #compute cost & gradients, update model weights
#             jw=mse(target,v_s)  #J(w) (for 1 sample)
#             jw.backward()       #compute gradients
#             optim.step()        #step the optimizer, update weights
#             optim.zero_grad()   #zero out gradients (otherwise PyTorch will keep accumulating gradients)
#
#             #check termination status from environment (reached terminal state?)
#             if term_status: break  #if termination status is True, we've reached end of episode
#
#             state=next_state
#             episode_length+=1
#
#         if episode%500 == 0:
#             print('\tEpisode: {}'.format(episode))
#
#
#     return


