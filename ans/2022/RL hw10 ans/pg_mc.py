#
# Policy-gradient with Monte Carlo (REINFORCE)
#
#

import torch 

#
#   Policy-gradient with Monte Carlo
#
def pg_mc(simenv, policy, gamma, alpha, num_episodes, max_episode_len, use_gamma_t=True, window_len=100, term_thresh=None, showPlots=False, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance
        policy:  Parameterized model for the policy (e.g. neural network)
        gamma :  Future discount factor, between 0 and 1
        alpha :  Learning step size
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        use_gamma_t:  If True, use gamma^t term in gradient update
        window_len: Window size for total rewards windowed average (averaged over multiple episodes)
        term_thresh: If windowed average > term_thresh, stop (if None, run until num_episodes)
        showPlots:  If True, show plot of episode lengths during training, default: False
        prng_seed:  Seed for the random number generator        
    '''

    #initialize a few things
    #
    simenv.reset(seed=prng_seed)

    optim=torch.optim.Adam(policy.parameters(),lr=alpha,maximize=True) #optimizer

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

        #initial state
        state=simenv.reset()

        #Run episode state-action-reward sequence to end
        #
        episode_length=0
        rewards=[]  #need to track rewards for each step in episode
        log_pis=[]  #need to track log pi terms for each step (note, includes PyTorch gradient tracking)
        while episode_length < max_episode_len:
            #choose action for current state, based on current policy
            (action, log_pi)=policy.choose_action(state)

            #take action, get reward, next state, termination status
            (next_state,reward,term_status,_)=simenv.step(action)
            tot_reward+=reward

            #track rewards, log_pi's
            rewards.append(reward)
            log_pis.append(log_pi)

            #check termination status from environment (reached terminal state?)
            if term_status: break  #if termination status is True, we've reached end of episode

            #move to next step in episode
            state=next_state
            episode_length+=1
        
        
        #compute discounted return for each visited state
        tot_return=0
        tot_returns=torch.zeros(len(rewards), dtype=torch.float32)
        for i in range(len(rewards)-1,-1,-1): #go backwards
            tot_return=tot_return*gamma+rewards[i]
            tot_returns[i]=tot_return
        
        #next, forward compute the additional gamma^t factor shown in Sutton's gradient update rule (pseudocode)
        # (note, it is not uncommon to see implementations that do not include this gamma^t term,
        #    for example, Mnih & Deepmind's 2016 A2C paper, "Asynchronous Methods for Deep Reinforcement Learning")
        if use_gamma_t:
            for i in range(len(tot_returns)):
                tot_returns[i]=(gamma**i)*tot_returns[i]
            
        #Compute gradient target function for entire episode,
        # then use built-in optimizer to apply desired gradient-based weight updates all at once
        logpi_stack=torch.stack(log_pis)
        gradtgt=torch.sum(tot_returns*logpi_stack)
        
        #apply the weight update
        #
        gradtgt.backward()
        optim.step()
        optim.zero_grad()
        
        
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
