#
# A brief demo & explanation of the Cat_and_Mouse simulation environment features
#

from cat_and_mouse import Cat_and_Mouse
import numpy as np

# local define
gamma = 0.9
episode_num_part1 = 500
episode_num_part2 = 50000
max_iter = 100


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
        if visit_dirty_bit[curr_state, action] == False:
            visit_dirty_bit[curr_state, action] = True
            first_visit = True
            epi_list.append((curr_state, action, reward, first_visit))
        else:
            epi_list.append((curr_state, action, reward, first_visit))
        if gameOver == True:
            break
        # define action
        action = policy[curr_state][0]
        # update current state
        curr_state = nextState
        # update iter
        iter += 1
    # cm.reset()
    epi_list.append(iter)
    return epi_list


def loop_episode(epi_list, gamma, Q, policy, visit_times):
    iter = epi_list.pop()
    G = 0
    while iter > 0:
        (St, At, R, first_visit) = epi_list.pop()
        G = gamma * G + R  # G = gamma*G + R(i+1)
        iter -= 1
        # unless the pair St, At appears in S0,A0.....
        if first_visit == True:
            visit_times[St, At] += 1
            # P21 first visit incremental update
            Q[St, At] = Q[St, At] + \
                (1.0/visit_times[St, At]) * (G - Q[St, At])
            # greedy policy update for this state
            policy[St] = np.argmax(Q[St])
        # elif np.all(visit_dirty_bit):  # == np.ones([cm.numStates])
        #    break
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
    visit_times = np.zeros([cm.numStates, cm.numActions])
    # print initial policy
    print("---- reshape policy1 ----")
    print(policy.reshape(cm.rows, cm.columns))
    for i in range(episode_num_part1):
        epi_list = episode(cm, policy, max_iter)
        policy, Q, visit_times = loop_episode(
            epi_list, gamma, Q, policy, visit_times)
        # if (i < 50):
        #   print(visit_times)
        #    print("the %d index is", i)
        #    print(policy.reshape(cm.rows, cm.columns))
        #   print(Q)
    # print final policy
    print(policy.reshape(cm.rows, cm.columns))
    # cm.policy2gif(policy, cm.mouseInitLoc, 'cm_demo1')
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
    print(policy2.reshape(cm.rows, cm.columns))
    for i in range(episode_num_part2):
        epi_list = episode(cm, policy2, max_iter)
        policy2, Q2, visit_times = loop_episode(
            epi_list, gamma, Q2, policy2, visit_times)
    # print final policy
    print("---- reshape policy2 ----")
    print(policy2.reshape(cm.rows, cm.columns))
    # cm.policy2gif(policy2, cm.mouseInitLoc, 'cm_demo2')
    print("---- Q2 ----")
    print(Q2)
