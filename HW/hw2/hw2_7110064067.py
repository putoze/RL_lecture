import matplotlib.pyplot as plt
import numpy as np

SIZE = 5
POS_A = [0, 1]
POS_A0 = [4, 1]
POS_B = [0, 3]
POS_B0 = [2, 3]
gamma = 0.9
probability = 0.25
actions = [np.array([-1, 0]),#up
           np.array([1, 0]), #down
           np.array([0, -1]), #left
           np.array([0, 1])] #right

def step(state, action):
    if state == POS_A:
        return POS_A0, 10
    if state == POS_B:
        return POS_B0, 5

    nstate = (np.array(state) + action).tolist()
    x, y = nstate
    if (x < 0) or (x >= SIZE) or (y < 0) or (y >= SIZE):
        reward = -1
        nstate = state
    else:
        reward = 0
    return nstate, reward

def draw_table(values):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax = plt.table(cellText=values,
                   cellLoc='center',
                   bbox=[0, 0, 1, 1])
    plt.show()

def calculate():
    value = np.zeros((SIZE, SIZE))
    while True:
        new_value = np.zeros_like(value)
        for i in range(SIZE):
            for j in range(SIZE):
                for action in actions:
                    (next_i, next_j), reward = step([i, j], action)
                    new_value[i, j] += probability * (reward + gamma * value[next_i, next_j]) #bellman equation
        
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_table(np.round(new_value, decimals=1))
            break
        value = new_value

if __name__ == '__main__':
    calculate()