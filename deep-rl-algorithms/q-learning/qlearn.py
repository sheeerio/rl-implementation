import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

ALPHA = 1e-3
GAMMA = 0.9
EPS_MAX = 1.0
EPS_MIN = 0.01
env  = gym.make('FrozenLake-v1')

class Agent():
    def __init__(self, lr, gamma, eps_start, eps_end, eps_dec):
        self.Q = np.zeros((env.observation_space.n, \
            env.action_space.n))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
    
    def update_epsilon(self):
        self.epsilon = self.epsilon*self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
    
    def learn(self, state, action, reward, state_):
        best_a = np.argmax(self.Q[state_])
        self.Q[state, action] += self.lr*(reward + \
            self.gamma*self.Q[state_, best_a]-self.Q[state, action])
        self.update_epsilon()

if __name__ == "__main__":
    agent = Agent(lr=ALPHA,gamma=GAMMA,eps_start=EPS_MAX,\
                    eps_end=EPS_MIN,eps_dec=0.9999995)
    scores = []
    wins = []
    for e in range(500000):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            state_, rew, done, *_ = env.step(action)
            agent.learn(state, action, rew, state_)
            score += rew
            state = state_
        scores.append(score)
        if e % 100 == 0:
            win = np.mean(scores[-100:])
            wins.append(win)
            if e % 1000 == 0:
                print('episode ', e, 'win pct %.2f' % win, \
                    'eps %.2f' % agent.epsilon)

plt.plot(wins)
plt.show()