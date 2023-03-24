#
# python ./hw4_7111064109.py
#

from cat_and_mouse import Cat_and_Mouse
import numpy as np

# local define
gamma = 0.9
episode_num_1 = 500
episode_num_2 = 5000
alpha = 0.1  # 0~1
np.set_printoptions(precision=3)


def choose_action(cm, Q, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(cm.numActions)
    else:
        st = cm.currentState()
        action = np.argmax(Q[st])
    return action


def Q_learning(cm, gamma, alpha, episode_num):
    epsilon0 = 1.0
    # initial Q
    Q = np.zeros((cm.numStates, cm.numActions), dtype=np.float32)
    # initial policy
    policy = np.zeros((cm.numStates, 1), dtype=np.int32)
    # loop for episode
    for n in range(episode_num):
        # update epsilon
        epsilon = epsilon0*(1-n/episode_num)
        # initial current state
        curr_state = cm.prng.randint(0, cm.numStates-1)
        cm.initState(curr_state)
        while True:
            action = choose_action(cm, Q, epsilon)
            (next_state, reward, gameOver) = cm.step(action)
            Q[curr_state, action] += alpha * \
                (reward + gamma*np.max(Q[next_state]) - Q[curr_state, action])
            curr_state = next_state
            if gameOver:
                break
    for st in range(cm.numStates):
        policy[st][0] = np.argmax(Q[st])
    return policy, Q


if __name__ == '__main__':
    print("------------ part (1) ---------------")
    # set environment
    cm = Cat_and_Mouse(rows=1, columns=7, mouseInitLoc=[0, 3], cheeseLocs=[
        [0, 0], [0, 6]], stickyLocs=[[0, 2]], slipperyLocs=[[0, 4]], prngSeed=8787)
    # ----------- Q_learning --------------
    policy, Q = Q_learning(cm, gamma, alpha, episode_num_1)
    # print final policy
    print("---- reshape final policy1 ----")
    print(policy.reshape(cm.rows, cm.columns))
    # Gif
    cm.policy2gif(policy, [0, 3], 'cm_demo1_A')
    cm.policy2gif(policy, [0, 2], 'cm_demo1_B')
    print("---- Q1 ----")
    print(Q)
    print("------------ part (2) ---------------")
    cm = Cat_and_Mouse(slipperyLocs=[[1, 1], [2, 1]], stickyLocs=[
        [2, 4], [3, 4]], catLocs=[[3, 2], [3, 3]])
    policy2, Q2 = Q_learning(cm, gamma, alpha, episode_num_2)
    print("---- reshape final policy1 ----")
    print(policy2.reshape(cm.rows, cm.columns))
    # Gif
    cm.policy2gif(policy2, [0, 0], 'cm_demo2_A')
    cm.policy2gif(policy2, [0, 4], 'cm_demo2_B')
    cm.policy2gif(policy2, [2, 2], 'cm_demo2_C')
    print("---- Q2 ----")
    print(Q2)
