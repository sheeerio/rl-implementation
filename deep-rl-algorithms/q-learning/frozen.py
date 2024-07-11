import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')
EPISODES = 1000

win = []
scores = []
for e in range(EPISODES):
    score = 0
    observation, _ = env.reset()
    done = False
    while not done:
        obs_, rew, done, truncated, _ = env.step(env.action_space.sample())
        score += rew
    scores.append(score)
    if e % 10 == 0:
        win.append(np.mean(scores[-10:]))
plt.plot(win)
plt.show()