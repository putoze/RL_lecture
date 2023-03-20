#
# A brief demo & explanation of the Cat_and_Mouse simulation environment features
#

from cat_and_mouse import Cat_and_Mouse
import numpy as np

# local define
gamma = 0.9
episode_num = 50000


def episode(cm, policy):
    epi_list = []
    iter = 0
    max_iter = 100
    # random init
    cm.initState(cm.prng.randint(0, cm.numStates-1))
    gameOver = False
    action = cm.prng.randrange(cm.numActions)
    while iter < max_iter:
        curr_state = cm.currentState()
        (nextState, reward, gameOver) = cm.step(action)
        epi_list.append((curr_state, action, reward))
        action = policy[curr_state][0]
        #print(curr_state, nextState)
        if gameOver == True:
            break
        iter += 1
    cm.reset()
    epi_list.append(iter)
    return epi_list, policy


def loop_episode(cm, epi_list, gamma, Q, policy, visit_times):
    iter = epi_list.pop()
    visit_dirty_bit = np.zeros([cm.numStates])
    G = 0
    while iter > 0:
        (St, At, R) = epi_list.pop()
        G = gamma * G + R  # G = gamma*G + R(i+1)
        iter -= 1
        if visit_dirty_bit[St] == 0:
            visit_dirty_bit[St] = 1
            visit_times[St, At] += 1
            # P43 first visit incremental update
            Q[St, At] = Q[St, At] + \
                (1/visit_times[St, At]) * (G - Q[St, At])
            # greedy policy update for this state
            policy[St] = np.argmax(Q[St])
        # elif np.all(visit_dirty_bit):  # == np.ones([cm.numStates])
        #    break

    return policy, Q, visit_times


if __name__ == '__main__':
    print("------------ part1 ---------------")
    cm = Cat_and_Mouse(rows=1, columns=7, mouseInitLoc=[0, 3], cheeseLocs=[
        [0, 0], [0, 6]], stickyLocs=[[0, 2]], slipperyLocs=[[0, 4]])
    # initial policy
    policy = np.random.randint(
        low=0, high=cm.numActions, size=(cm.numStates, 1), dtype=np.int32)
    # initial Q
    Q = np.zeros([cm.numStates, cm.numActions])
    # initial visit_times
    visit_times = np.zeros([cm.numStates, cm.numActions])
    # print initial policy
    print(policy.reshape(cm.rows, cm.columns))
    for i in range(episode_num):
        epi_list, policy = episode(cm, policy)
        policy, Q, visit_times = loop_episode(
            cm, epi_list, gamma, Q, policy, visit_times)
    # print final policy
    print(policy.reshape(cm.rows, cm.columns))
    cm.policy2gif(policy, cm.mouseInitLoc, 'cm_demo1')
    print(Q)

    print("------------ part2s ---------------")
    cm = Cat_and_Mouse(slipperyLocs=[[1, 1], [2, 1]], stickyLocs=[
        [2, 4], [3, 4]], catLocs=[[3, 2], [3, 3]])
    # initial policy
    policy2 = np.random.randint(
        low=0, high=cm.numActions, size=(cm.numStates, 1), dtype=np.int32)
    # initial Q
    Q2 = np.zeros([cm.numStates, cm.numActions])
    # initial visit_times
    visit_times = np.zeros([cm.numStates, cm.numActions])
    # print initial policy
    print(policy2.reshape(cm.rows, cm.columns))
    for i in range(episode_num):
        epi_list, policy2 = episode(cm, policy2)
        policy2, Q2, visit_times = loop_episode(
            cm, epi_list, gamma, Q2, policy2, visit_times)
    # print final policy
    print(policy2.reshape(cm.rows, cm.columns))
    cm.policy2gif(policy2, cm.mouseInitLoc, 'cm_demo2')
    print(Q2)
