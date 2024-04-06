import numpy as np
import random

class GaussianStationaryBandit(object):
    def __init__(self, k, mu, sigma):
        self.qstar = np.array(mu)           # ndarray of optimal expected payoff
        self.sigma = np.array(sigma)        # ndarray of optimal std. of payoff distribution
        self.history = np.zeros(k)          # intialize history of all arms taken
        self.regret = []                    # intialize history of regret of each action
        self.opt_q = np.max(self.qstar)
        self.opt_arm = np.argmax(self.qstar)
        self.num_arms = k
        self.reward = 0

    def pull(self, arm):
        self.history[arm] += 1
        self.regret.append(self.opt_q - self.qstar[arm])
        reward = random.guass(self.qstar[arm], self.sigma[arm])
        self.reward += reward
        return reward
    
    def get_opt_arm(self):
        return self.opt_arm
    
    def get_history(self):
        return self.history

    def get_regret(self):
        return self.regret

    def reset(self):
        self.history = np.zeros(self.num_arms)
        self.regret = []
        self.reward = 0

    def get_total_reward(self):
        return self.reward


class BernoulliStationaryBandit(object):
    def __init__(self, k, mu):
        self.qstar = np.array(mu)
        self.history = np.zeros(k)
        self.regret = []
        self.reward = 0
        self.opt_q = np.max(self.qstar)
        self.opt_arm = np.argmax(self.qstar)
        self.num_arms = k

    def pull(self, arm):
        self.history[arm] += 1
        self.regret.append(self.opt_q - self.qstar[arm])
        reward = np.random.choice([1,0], p=[self.qstar[arm], 1-self.qstar[arm]])
        self.reward += reward
        return reward
    
    def get_opt_arm(self):
        return self.opt_arm
    
    def get_history(self):
        return self.history

    def get_regret(self):
        return self.regret

    def reset(self):
        self.history = np.zeros(self.num_arms)
        self.regret = []
        self.reward = 0
    
    def get_total_reward(self):
        return self.reward