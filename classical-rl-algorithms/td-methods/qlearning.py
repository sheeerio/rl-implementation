from environment import ENVIRONMENT
from sarsa import random_policy
import numpy as np
import cv2

env = ENVIRONMENT(diagonal=True, size=10, num_enemy = 3, num_food = 1)
episodes = 25000
nS = 100
nA = 4
learning_rate = 0.01
gamma = 0.9
epsilon = 0.95

Q = np.zeros([nS, nA])
for e in range(episodes):
    pos_i,rew,n_term = env.reset(newpos=True)
    s_i = pos_i[0]*10 + pos_i[1]
    a_i = random_policy(Q, s_i, epsilon)
    while n_term == False:
        pos_f,(rf,term) = env.step(a_i)
        s_f = pos_f[0]*10 + pos_f[1]
        a_f = random_policy(Q, s_f, epsilon)
        Q[s_i,a_i] = (1-learning_rate)*Q[s_i,a_i] + learning_rate*(rf + gamma*np.max(Q[s_f]))
        a_i = a_f
        s_i = s_f 
    epsilon *= 1-e-4

pol = np.zeros(nS)
for s in range(nS):
    pol[s] = np.argmax(Q[s])

def play():
    pos_i,k,ter = env.startover(newpos=True)
    env.render()
    T = False
    i = 0
    while T == False and i<=20:   
        s = pos_i[0]*10 + pos_i[1]
        #print(pos_i)
        pos_i,R = env.step(pol[s])
        #print(R)
        T=R[1]
        env.render(500)
        i = i+1
    cv2.destroyAllWindows()

for i in range(20):
    play()    