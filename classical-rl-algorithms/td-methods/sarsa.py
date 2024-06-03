import random
import numpy as np
from environment import ENVIRONMENT


env = ENVIRONMENT(diagonal=True, size=10, num_enemy = 3, num_food = 1)
episodes = 25000
nS = 100
nA = 4
learning_rate = 0.01
gamma = 0.9
epsilon = 0.95

def random_policy(Q, s, epsilon):
    r = random.random()
    if r < epsilon:
        a = env.sample_action()
    else:
        a = np.argmax(Q[s])
    return a