import random
import numpy as np
from bandits import BernoulliStationaryBandit
from bandits import GaussianStationaryBandit

class epsilon_greedy_agent(BernoulliStationaryBandit):
    def __init__(self, bandit, epsilon, num_iters, decay_factor):
        self.bandit = bandit
        self.epsilon = epsilon
        self.num_iters = num_iters
        self.Q = np.ones(self.bandit.num_arms)
        self.decay_factor = decay_factor
    
    def Policy(self):
        if random.random()<self.epsilon:
            arm = random.randint(0, self.bandit.num_arms-1)
        else:
            arm = np.argmax(self.Q)
        return arm
    
    def play(self, decay=True):
        self.bandit.reset()
        self.indicator = np.zeros(self.bandit.num_arms)
        for t in range(self.num_iters):
            arm = self.Policy()
            reward = self.bandit.pull(arm)
            self.indicator[arm] += 1
            self.Q[arm] += (reward - self.Q[arm])/(self.indicator[arm]+1)
            if decay and t>0:
                if t > self.num_iters/self.decay_factor:
                    self.epsilon = 1/t
        arm_history = self.bandit.get_history()
        regret = self.bandit.get_regret()
        total_reward = self.bandit.get_total_reward()
        return self.Q, arm_history, regret, total_reward

class softmax_agent(BernoulliStationaryBandit):
    def __init__(self, bandit, beta, num_iters):
        self.bandit = bandit
        self.beta = beta
        self.num_iters = num_iters
        self.Q = np.ones(self.bandit.num_arms)
        self.indicator = np.zeros(self.bandit.num_arms)

    def Policy(self):
        prob = np.copy(np.exp(self.Q/self.beta)/np.sum(np.exp(self.Q/self.beta)))
        arm = np.random.choice(np.arange(self.bandit.num_arms), p=prob)
        return arm
    
    def play(self):
        self.bandit.reset()
        for t in range(self.num_iters):
            arm = self.Policy()
            reward = self.bandit.pull(arm)
            self.indicator[arm]+=1
            self.Q[arm] += (reward - self.Q[arm])/(self.indicator[arm]+1)
        arm_history = self.bandit.get_history()
        regret = self.bandit.get_regret()
        total_reward = self.bandit.get_total_reward()
        return self.Q, arm_history, regret, total_reward

class UCB_agent(BernoulliStationaryBandit):
    def __init__(self, bandit, num_iters):
        self.bandit = bandit
        self.num_iters = num_iters
        self.time = 0
        self.Q = np.zeros(self.bandit.num_arms)
        self.confidence = np.zeros(self.bandit.num_arms)
    
    def Policy(self):
        arm = np.argmax(np.add(self.Q, self.confidence))
        return arm

    def play(self):
        self.bandit.reset()
        for i in range(self.bandit.num_arms):
            self.Q[i] = self.bandit.pull(i)
            self.time += 1
        for t in range(self.num_iters - self.bandit.num_arms):
            ah = self.bandit.get_history()
            self.confidence = np.sqrt(2*np.log(self.time+t+1)/ah)
            arm = self.Policy()
            reward = self.bandit.pull(arm)
            self.Q[arm] += (reward - self.Q[arm])/(ah[arm]+1)
        arm_history = self.bandit.get_history()
        regret = self.bandit.get_regret()
        total_reward = self.bandit.get_total_reward()
        return self.Q, arm_history, regret, total_reward

class Median_Elimination_Agent(BernoulliStationaryBandit):
    def __init__(self, bandit, epsilon, delta):
        self.bandit = bandit
        self.epsilon = epsilon/4
        self.delta = delta/2
        self.Q = np.ones(self.bandit.num_arms)
        self.S = np.arange(self.bandit.num_arms)
        self.indicator = np.zeros(self.bandit.num_arms)

    def play(self):
        self.bandit.reset()
        while len(self.S) != 1:
            for arm in self.S:
                count = int(2*np.log(3/self.delta)/np.square(self.epsilon))
                for i in range(count):
                    reward = self.bandit.pull(arm)
                    self.indicator[arm] += 1
                    self.Q[arm] += (reward-self.Q[arm])/(self.indicator[arm]+1)
            M = np.median(self.Q[self.S])
            self.S = np.delete(self.S, np.where(self.Q[self.S] < M))
            self.epsilon *= 3/4
            self.delta /= 2
            print(self.S)
        arm_history = self.bandit.get_history()
        regret = self.bandit.get_regret()
        total_reward = self.bandit.get_total_reward()
        return self.S, arm_history, regret, total_reward

class Thompson_Sampling_Agent(BernoulliStationaryBandit):
    def __init__(self, bandit, alpha, beta, num_iters):
        self.bandit = bandit
        self.alpha = alpha
        self.beta = beta
        self.num_iters = num_iters
        self.indicator = np.zeros(self.bandit.num_arms)
        self.Q = np.zeros(self.bandit.num_arms)
    
    def Policy(self):
        samples = np.random.normal(loc = self.Q, scale = self.alpha/(np.sqrt(self.indicator)+self.beta))
        arm = np.argmax(samples)
        return arm
    
    def play(self):
        self.bandit.reset()
        for t in range(self.num_iters):
            arm = self.Policy()
            reward = self.bandit.pull(arm)
            self.indicator[arm]+=1
            self.Q[arm] += (reward-self.Q[arm])/(self.indicator[arm]+1)
        arm_history = self.bandit.get_history()
        regret = self.bandit.get_regret()
        total_reward = self.bandit.get_total_reward()
        return self.Q, arm_history, regret, total_reward

