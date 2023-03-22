#
# python ./hw4_7111064109.py
#

from cat_and_mouse import Cat_and_Mouse
import numpy as np

# local define
gamma = 0.9
episode_num_part1 = 5000
episode_num_part2 = 50000
max_iter = 100
np.set_printoptions(precision=3)


def episode(cm, policy, max_iter):
    epi_list = []
    iter = 0
    # choose S0 and A0 randomly
    curr_state = cm.prng.randint(0, cm.numStates-1)
    cm.initState(curr_state)
    action = cm.prng.randrange(0, cm.numActions, 1)
    # set visit dirty bit
    visit_dirty_bit = np.zeros((cm.numStates, cm.numActions), dtype=bool)
    # avoid infinite loop
    while iter < max_iter:
        (nextState, reward, gameOver) = cm.step(action)
        first_visit = False
        # define epi_list
        if not visit_dirty_bit[curr_state, action]:
            visit_dirty_bit[curr_state, action] = True
            first_visit = True
        # append epi_list
        epi_list.append((curr_state, action, reward, first_visit))
        # check if game over
        if gameOver:
            break
        # update current state
        curr_state = nextState
        # define action
        action = policy[curr_state][0]
        # update iter
        iter += 1
    # cm.reset()
    return epi_list


def loop_episode(epi_list, gamma, Q, policy, visit_times):
    G = 0
    while len(epi_list) > 0:  # check if empty epi_list
        (St, At, R, first_visit) = epi_list.pop()
        G = gamma * G + R  # G = gamma*G + R(i+1)
        # unless the pair St, At appears in S0,A0.....
        if first_visit:
            visit_times[St, At] += 1
            # P21 first visit incremental update
            Q[St, At] += (1.0/visit_times[St, At]) * (G - Q[St, At])
            # greedy policy update for this state
            policy[St] = np.argmax(Q[St])
    return policy, Q, visit_times


if __name__ == '__main__':
    print("------------ part1 ---------------")
    # set environment
    cm = Cat_and_Mouse(rows=1, columns=7, mouseInitLoc=[0, 3], cheeseLocs=[
        [0, 0], [0, 6]], stickyLocs=[[0, 2]], slipperyLocs=[[0, 4]], prngSeed=8787)
    # ----------- initial value --------------
    # initial policy
    policy = np.random.randint(
        cm.numActions, size=(cm.numStates, 1), dtype=np.int32)
    # initial Q
    Q = np.zeros((cm.numStates, cm.numActions), dtype=np.float32)
    # initial visit_times
    visit_times = np.zeros((cm.numStates, cm.numActions), dtype=np.int32)
    # print initial policy
    print("---- reshape initial policy1 ----")
    print(policy.reshape(cm.rows, cm.columns))
    # loop forever
    for i in range(episode_num_part1):
        epi_list = episode(cm, policy, max_iter)
        policy, Q, visit_times = loop_episode(
            epi_list, gamma, Q, policy, visit_times)
    # print final policy
    print("---- reshape final policy1 ----")
    print(policy.reshape(cm.rows, cm.columns))
    # Gif
    cm.policy2gif(policy, [0, 3], 'cm_demo1_A')
    cm.policy2gif(policy, [0, 2], 'cm_demo1_B')
    print("---- Q1 ----")
    print(Q)
    print("------------ part2s ---------------")
    # set environment
    cm = Cat_and_Mouse(slipperyLocs=[[1, 1], [2, 1]], stickyLocs=[
        [2, 4], [3, 4]], catLocs=[[3, 2], [3, 3]])
    # ----------- initial value --------------
    # initial policy
    policy2 = np.random.randint(
        low=0, high=cm.numActions, size=(cm.numStates, 1), dtype=np.int32)
    # initial Q
    Q2 = np.zeros([cm.numStates, cm.numActions])
    # initial visit_times
    visit_times = np.zeros([cm.numStates, cm.numActions])
    # print initial policy
    print("---- reshape initial policy2 ----")
    print(policy2.reshape(cm.rows, cm.columns))
    # loop forever
    for i in range(episode_num_part2):
        epi_list = episode(cm, policy2, max_iter)
        policy2, Q2, visit_times = loop_episode(
            epi_list, gamma, Q2, policy2, visit_times)
    # print final policy
    print("---- reshape final policy2 ----")
    print(policy2.reshape(cm.rows, cm.columns))
    # Gif
    cm.policy2gif(policy2, [0, 0], 'cm_demo2_A')
    cm.policy2gif(policy2, [0, 4], 'cm_demo2_B')
    cm.policy2gif(policy2, [2, 2], 'cm_demo2_C')
    print("---- Q2 ----")
    print(Q2)
