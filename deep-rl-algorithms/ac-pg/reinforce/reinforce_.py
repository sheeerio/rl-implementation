import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'mps')
        self.to(self.device)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent():
    def __init__(self, lr, input_dims, n_actions=4, gamma=0.99):
        self.gamma = gamma
        self.network = PolicyNetwork(lr, input_dims, n_actions)
        self.rewards = []
        self.lgprobs = []
    
    def choose_action(self, obs):
        obs = T.tensor(obs, dtype=T.float).to(self.network.device)
        probs = F.softmax(self.network(obs))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.lgprobs.append(log_probs)
        
        return action.item()

    def store_rewards(self, reward):
        self.rewards.append(reward)
    
    def learn(self):
        self.network.optimizer.zero_grad()

        G = np.zeros_like(self.rewards, dtype=np.float32)
        for t in range(len(self.rewards)):
            sum = 0
            discount = 1
            for t_ in range(t, len(self.rewards)):
                sum += discount * self.rewards[t_]
                discount *= self.gamma
            G[t] = sum
        G = T.tensor(G, dtype=T.float).to(self.network.device)

        loss = 0
        for g, lgprob in zip(G, self.lgprobs):
            loss += -g * lgprob
        loss.backward()
        self.network.optimizer.step()

        self.rewards = []
        self.lgprobs = []