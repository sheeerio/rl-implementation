import numpy as np
from environment import OGEnvironment

env = OGEnvironment()
GAMMA = 0.9
EPSILON = 0.01

## value iteration and policy iteration
def value_iteration():
    v = np.zeros([5, 5])
    v_new = np.zeros([5, 5])
    while True:
        for y in range(5):
            for x in range(5):
                v_temp = np.zeros(4)
                for action in range(4):
                    env.x = x
                    env.y = y
                    x_next, y_next, reward = env.step(action)
                    v_temp[action] = reward + GAMMA*v[y_next, x_next]
                v_new[y,x] = np.max(v_temp)
        if np.max(np.abs(v - v_new)) < EPSILON*(1-GAMMA)/(2*GAMMA):
            env.plot_grid_values(np.round(v_new,decimals=2))
            break
        v = np.copy(v_new)

def policy_iteration():
    n = 0
    policy = np.zeros([5,5],dtype = np.uint8)
    v = np.zeros([5,5])
    v_new = np.zeros([5,5])
    while True:
        #Policy evaluation
        while True:
            for y in range(5):
                for x in range(5):
                    action = policy[y,x]
                    env.x = x
                    env.y = y
                    x_next,y_next,reward = env.step(action)
                    v_new[y,x] = reward + GAMMA*v[y_next,x_next]
            if np.max(np.abs(v - v_new)) < EPSILON*(1-GAMMA)/(2*GAMMA):
                break
            v = np.copy(v_new)
        #Policy improvement
        new_policy = np.zeros([5,5],dtype=np.uint8)
        for y in range(5):
            for x in range(5):
                v_temp = np.zeros(4)
                for action in range(4):
                    env.x = x
                    env.y = y
                    x_next,y_next,reward = env.step(action)
                    v_temp[action] = reward + GAMMA*v[y_next,x_next]
                new_policy[y,x] = np.argmax(v_temp)
        if np.array_equal(policy,new_policy):
            break
        policy = np.copy(new_policy)
    env.plot_policy(policy)
    return policy


def Play_optimally(policy):
    (x,y) = env.reset()
    for i in range(25):
        action = policy[y,x]
        (x,y,reward) = env.step(action)
        print(reward)
        env.render()


value_iteration()

policy = policy_iteration()
Play_optimally(policy)