import random
import numpy as np
from environment import ENVIRONMENT, Blob


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

Q = np.zeros([nS, nA])
for e in range(episodes):
    pos_i,rew,n_term = env.reset(newpos=True)
    s_i = pos_i[0]*10 + pos_i[1]
    a_i = random_policy(Q, s_i, epsilon)
    while n_term == False:
        pos_f,(rf,term) = env.step(a_i)
        s_f = pos_f[0]*10 + pos_f[1]
        a_f = random_policy(Q, s_f, epsilon)
        Q[s_i,a_i] = (1-learning_rate)*Q[s_i,a_i] + learning_rate*(rf + gamma*Q[s_f,a_f])
        a_i = a_f
        s_i = s_f 
    epsilon *= 1-e-4