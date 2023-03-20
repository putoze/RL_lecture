from cat_and_mouse import Cat_and_Mouse
from random import Random
import numpy as np


def EPISODE(CM, PROG):
    X = PROG.choice(list(range(CM.rows)))
    Y = PROG.choice(list(range(CM.columns)))
    episode = []

    CM.initState([X, Y])

    iterate = 0
    gameOver = False
    while not gameOver:
        S = CM.currentState()
        A = PROG.choice(list(CM.actions))
        (nextState, reward, gameOver) = CM.step(A)
        R = reward
        episode.append((S, A, R))
        iterate += 1
    CM.reset()
    return episode


def CM_policy(CM, PROG, GAMMA, ITERATE, DIM):
    policy = np.zeros((CM.numStates, 1), dtype=np.int32)
    Q = np.zeros([CM.numStates, CM.numActions])
    RT_SA = np.zeros([CM.numStates, CM.numActions])
    RT_C = np.zeros([CM.numStates, CM.numActions])

    for i in range(ITERATE):
        episode = EPISODE(CM, PROG)
        G = 0
        for j in range(1, len(episode) + 1):
            G = GAMMA*G + episode[-j][2]
            if episode[-j][0:2] not in episode[:-j][0:2]:
                s = episode[-j][0]
                a = episode[-j][1]
                r = episode[-j][2]

                RT_SA[s][a] += G
                RT_C[s][a] += 1
                Q[s][a] = RT_SA[s][a] / RT_C[s][a]

                if (DIM == 2):
                    t_i = Q[s].argsort()[::-1][0:CM.numActions]
                    num = 0
                    X = s // CM.columns
                    Y = s % CM.columns
                    CM.initState([X, Y])
                    (n, _, _) = CM.step(t_i[num])
                    while ((n < s) or (n % 5 < Y)):
                        num += 1
                        CM.initState([X, Y])
                        (n, _, _) = CM.step(t_i[num])
                    policy[s][0] = t_i[num]
                else:
                    policy[s][0] = np.argmax(Q[s])
    return Q, policy


if __name__ == '__main__':
    PROG = Random()
    PROG.seed(456)
    GAMMA = 0.9

    ITERATE_1D = 50
    ITERATE_2D = 50000

    # cm1
    CM1D = Cat_and_Mouse(rows=1,
                         columns=7,
                         mouseInitLoc=[0, 3],
                         cheeseLocs=[[0, 0], [0, 6]],
                         stickyLocs=[[0, 2]],
                         slipperyLocs=[[0, 4]])
    Q1, policy1 = CM_policy(CM1D, PROG, GAMMA, ITERATE_1D, 1)
    #CM1D.policy2gif(policy1, CM1D.mouseInitLoc, 'CM1D')
    print(Q1)
    print(policy1)

    # cm2
    # cm2 will take some time
    CM2D = Cat_and_Mouse(rows=5,
                         columns=5,
                         mouseInitLoc=[0, 0],
                         cheeseLocs=[[4, 4]],
                         slipperyLocs=[[1, 1], [2, 1]],
                         stickyLocs=[[2, 4], [3, 4]],
                         catLocs=[[3, 2], [3, 3]])
    Q2, policy2 = CM_policy(CM2D, PROG, GAMMA, ITERATE_2D, 2)
    #CM2D.policy2gif(policy2, CM2D.mouseInitLoc, 'CM2D')
    print(Q2)
    print(policy2)
