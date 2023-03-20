#
# First-visit Monte-Carlo control, for estimating the optimal policy & q-function
#   q(s,a) [matrix: numStates x numActions], policy, a=pi(s) [column vector: numStates x 1]
#
#  Implementation: First-visit exploring-starts Monte Carlo control
#

from random import Random
import numpy as np

#
#  First-visit exploring-starts Monte Carlo control
#
#
#   Note: Since policy improvement iteration results in policies that are deterministic, we
#         will represent the policy, pi, here as a 1-D column vector, where each entry is an action index
#


def mcc_q(simenv, gamma, num_episodes, max_episode_len, printqPeriod=None, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance (e.g., Cat_and_Mouse)
        gamma :  Future discount factor, between 0 and 1
        num_episodes: Number of Monte-Carlo episodes to run
        max_episode_len: Maximum allowed length of a single episode
        printqPeriod:  Number of episodes between q(s,a) printouts.  If None, no q(s,a) printouts during run.
        prng_seed:  Seed for the random number generator

    Return value:
        The final value of pi(s) & q(s,a), the estimated optimal policy & q functions
    '''
    # initialize a few things
    #
    prng = Random()
    prng.seed(prng_seed)

    q = np.zeros((simenv.numStates, simenv.numActions),
                 dtype=np.float32)  # the q-function
    # total number of first visits per (state, action) across all episodes
    tot_first_visits = np.zeros(
        (simenv.numStates, simenv.numActions), dtype=np.int32)
    # assume number of actions is same for all states
    actions = list(range(simenv.numActions))

    # random initial policy
    policy = np.zeros((simenv.numStates, 1), dtype=np.int32)
    for i in range(simenv.numStates):
        policy[i] = prng.randint(0, simenv.numActions-1)

    # Start episode loop
    #
    for episode in range(num_episodes):
        # state,action,reward,first-visit-status sequence of an episode (s,a,r,firstVisit) tuple
        episode_rewards = []

        # using exploring starts, choose random init state & random init action
        #
        current_state = prng.randint(0, simenv.numStates-1)
        simenv.initState(current_state)
        action = prng.choice(actions)

        # Run episode state-action-reward sequence to end
        #
        episode_length = 0
        visited = np.zeros((simenv.numStates, simenv.numActions), dtype=bool)
        while episode_length < max_episode_len:
            # take action, get reward, next state, termination status
            (next_state, reward, term_status) = simenv.step(action)

            # update state-action-reward-visit sequence
            #
            first_visit = False
            if not visited[current_state, action]:
                first_visit = True
            visited[current_state, action] = True

            episode_rewards.append(
                (current_state, action, reward, first_visit))

            # check termination status from environment (reached terminal state?)
            if term_status:
                break  # if termination status is True, we've reached end of episode

            current_state = next_state
            action = policy[current_state][0]  # policy action for next step
            episode_length += 1

        # Process episode returns and q-function averages, greedy policy updates,
        #  working from end of episode to beginning
        #
        tot_return = 0
        for event in reversed(episode_rewards):
            (state, action, reward, first_visit) = event

            tot_return = tot_return*gamma+reward

            # if this event corresponds to first-visit to state/action, update q(s,a) average & greedy policy pi(s)
            if first_visit:
                tot_first_visits[state, action] += 1
                q[state, action] = q[state, action]+(1.0/tot_first_visits[state, action])*(
                    tot_return-q[state, action])  # incremental update of q(s,a) averages

                # greedy policy update for this state
                policy[state] = np.argmax(q[state])

        # if requested, print updated q estimates periodically
        #
        if (printqPeriod is not None) and (episode % printqPeriod == 0):
            print('Episode: {}'.format(episode))
            print(q)
            print('%%%%%%\n')

    return (policy, q)
