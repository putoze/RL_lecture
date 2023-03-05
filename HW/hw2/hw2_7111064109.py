import matplotlib.pyplot as plt
from numpy.linalg import inv
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
    #print(x, y)
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
    plt.title('Bellman Equation')
    plt.savefig('Bellman_Equation.png')
    plt.show()


def cal(P_matrix, R_matrix):
    location = np.array([0, 0])
    while not (location[1] == 5):
        i, j = location
        mat_idx = 5*j + i
        for action in action_list:
            next_state, reward = step(location, action)
            x, y = next_state
            nxt_idx = 5*y + x
            R_matrix[mat_idx] += reward*probability
            P_matrix[mat_idx][nxt_idx] += probability
        # print action P,R matrix
        # print("R_matrix")
        # print(R_matrix)
        # print(mat_idx)
        # print("P_matrix")
        # print(P_matrix[mat_idx].reshape(5, 5))
        # x,y index
        if (i == 4):
            location[0] = 0
            location[1] += 1
        else:
            location[0] += 1
    print("COMPLETE")
    # print("R_matrix")
    # print(R_matrix)
    # print("P_matrix")
    # print(P_matrix)


if __name__ == '__main__':
    P_matrix, R_matrix = create_matrix()
    cal(P_matrix, R_matrix)
    I_matrix = np.eye(SIZE, dtype='float32')
    V_matrix = np.dot(inv(I_matrix - gamma*P_matrix), R_matrix)
    V_matrix_round = np.round(V_matrix, decimals=1)
    draw_table(V_matrix_round.reshape(5, 5))
    print('V_matrix_round')
    print(V_matrix_round.reshape(5, 5))
    print('R_matrix')
    print(R_matrix.reshape(5, 5))
