import matplotlib.pyplot as plt
import numpy as np

# local define
SIZE = 25
POS_A = [1, 0]
POS_A_GOAL = [1, 4]
POS_B = [3, 0]
POS_B_GOAL = [3, 2]
gamma = 0.9
probability = 0.25
action_list = [np.array([0, -1]),  # up
               np.array([0,  1]),  # down
               np.array([-1, 0]),  # left
               np.array([1,  0])]  # right
actions_flags = ['↑',  '↓', '←', '→']


def create_matrix():
    P_matrix = np.zeros((SIZE, SIZE))
    R_matrix = np.zeros((SIZE, 1))
    return P_matrix, R_matrix


def step(location, action):
    if location.tolist() == POS_A:
        return POS_A_GOAL, 10
    if location.tolist() == POS_B:
        return POS_B_GOAL, 5
    next_state = location + action
    x, y = next_state
    # print(x, y)
    if (x < 0) or (x >= 5) or (y < 0) or (y >= 5):
        reward = -1
        next_state = location
    else:
        reward = 0
    return next_state, reward


def draw_table(V_matrix):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax = plt.table(cellText=V_matrix,
                   cellLoc='center',
                   bbox=[0, 0, 1, 1])
    plt.title('value function')
    plt.show()


def draw_policy(val_space):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax = plt.table(cellText=val_space,
                   cellLoc='center',
                   bbox=[0, 0, 1, 1])
    plt.title('direct matrix solution')
    plt.show()


def cal(P_matrix, R_matrix):
    for j in range(5):
        for i in range(5):
            mat_idx = 5*j + i
            for action in action_list:
                (x, y), reward = step(np.array([i, j]), action)
                nxt_idx = 5*y + x
                R_matrix[mat_idx] += reward*probability
                P_matrix[mat_idx][nxt_idx] += probability


'''
Textbook P37
1. Start with a policy (any policy)
2. Compute the value function or q function
3. Generate a new better (or equal) policy by acting greedily on (see next slide)
4. Iterate from step 2, as necessary. Converges towards
'''


def cal_itr(V_matrix):
    # Iterate from step 2
    while True:
        V_matrix_final = np.zeros_like(V_matrix)
        val_space = np.zeros((5, 5)).astype('str')  # string type
        for j in range(5):
            for i in range(5):
                mat_idx = 5*j + i  # location
                value_max = -999  # initial minimum value
                idx = 0
                for action in action_list:
                    (x, y), reward = step(np.array([i, j]), action)
                    nxt_idx = 5*y + x
                    # Compute the value
                    value = reward + gamma * V_matrix[nxt_idx]
                    # Generate a new better (or equal) policy
                    if value > value_max:
                        value_max = value
                        val_space[(j, i)] = actions_flags[idx]
                    elif value == value_max:
                        val_space[(j, i)] += actions_flags[idx]
                    # update action idx
                    idx += 1

                # optimal value function
                V_matrix_final[mat_idx] = value_max
        # test if Converge or not
        if np.max(np.abs(V_matrix_final - V_matrix)) < 1e-4:
            return V_matrix_final, val_space
        # update V_matrix
        V_matrix = V_matrix_final


if __name__ == '__main__':
    # create P_matrix,R_matrix
    P_matrix, R_matrix = create_matrix()
    cal(P_matrix, R_matrix)
    # create random V_matrix
    V_matrix = np.random.rand(SIZE, 1)
    # ------ HW3 part1 synchronous iterative ------
    delta = 0
    while True:
        V_matrix_tmp = V_matrix
        V_matrix = R_matrix + gamma*np.dot(P_matrix, V_matrix_tmp)
        delta = np.max(np.abs(V_matrix - V_matrix_tmp))
        # delta need to be small enough
        if delta < 1e-4:
            break
    # rounding
    V_matrix_round = np.round(V_matrix, decimals=1)
    # display
    draw_table(V_matrix_round.reshape(5, 5))
    print('V_matrix_round,HW3 Part1')
    print(V_matrix_round.reshape(5, 5))
    # ------ HW3 part2 Optimal Value function ------
    V_matrix, val_space = cal_itr(V_matrix)
    # rounding
    V_matrix_round = np.round(V_matrix, decimals=1)
    # display
    draw_table(V_matrix_round.reshape(5, 5))
    print('V_matrix_round,HW3 Part2')
    print(V_matrix_round.reshape(5, 5))
    print('direct matrix solution')
    print(val_space)
    draw_policy(val_space)
    print("COMPLETE")
