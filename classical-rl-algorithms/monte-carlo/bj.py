import gymnasium as gym
import numpy as np
import random
from utils import plot

env = gym.make('Blackjack-v1', natural=False)
GAMMA = 1
EPISODES = 200000

pSum = list(np.arange(4, 22))
aCard = list(np.arange(1, 11))
pAce = [True, False]
action_space = [0, 1]
state_space = []

target_policy = {}
Q = {}
C = {}

# intialize Q,C, target policy (pi)
for p in pSum:
    for a in aCard:
        for ace in pAce:
            state_space.append((p, a, ace))
            max = -1
            for action in action_space:
                Q[(p,a,ace),action] = random.random()
                if Q[(p,a,ace),action] > max:
                    max = Q[(p,a,ace),action]
                    argmax = action
                C[(p,a,ace),action] = 0
            target_policy[(p,a,ace)] = argmax

# define the behavior policy (b)
def behavior_policy():
    return np.random.choice(2, 1)[0]

# off-policy control using monte carlo rollouts
for _ in range(EPISODES):
    states, actions, rewards = [], [], []
    done = False
    s0, *_ = env.reset()
    states.append(s0)
    a = behavior_policy()
    actions.append(a)
    while True:
        s, r, done, *_ = env.step(a)
        rewards.append(r)
        if done:
            break
        states.append(s)
        actions.append(behavior_policy())
    G = 0
    W = 1
    for i in range(len(states)):
        G = GAMMA*G + rewards[-1-i]
        C[states[-1-i], actions[-1-i]] += W
        Q[states[-1-i],actions[-1-i]] += (G - Q[states[-1-i],actions[-1-i]])*(W/C[states[-1-i],actions[-1-i]])
        max = -1
        for action in action_space:
            if Q[states[-1-i],action] > max:
                max = Q[states[-1-i],action]
                argmax = action
        target_policy[states[-1-i]] = argmax
        if actions[-1-i] != argmax:
            break
        W *= (1/0.5)
    
def run(n):
    win, loss, draw = 0,0,0
    for _ in range(n):
        score = 0
        done = False
        s, *_ = env.reset()
        while not done:
            a = target_policy[s]
            s,r,done,*_ = env.step(a)
            score += r
        if score == 0: draw+=1
        elif score == 1: win += 1
        else: loss += 1
    print(f"wins: {win}, losses: {loss}, draws: {draw}\n")
    
plot(target_policy)
run(50)