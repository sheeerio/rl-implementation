import os

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2

def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, name, lr, hidden_dim=256, 
    chkpt_dir="../tmp/sac"):
        super(ValueNetwork, self).__init__()
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.hidden1 = nn.Linear(*state_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("mps")
        self.to(self.device)

    def forward(self, obs):
        x = F.relu(self.hidden1(obs))
        x = F.relu(self.hidden2(x))
        x = self.value(x)
        return x
    
    def save_checkpoint(self):
        print('... saving checkoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, name, lr, hidden_dims=256, 
    chkpt_dir="../tmp/sac"):
        super(QNetwork, self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.hidden1 = nn.Linear(state_dim[0] + n_actions, hidden_dims)
        self.hidden2 = nn.Linear(hidden_dims, hidden_dims)
        self.q_value = nn.Linear(hidden_dims, 1)
        self.apply(weights_init_)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("mps")
        self.to(self.device)

    def forward(self, obs, action):
        x = F.relu(self.hidden1(T.cat([obs, action], dim=1)))
        x = F.relu(self.hidden2(x))
        x = self.q_value(x)
        return x
    
    def save_checkpoint(self):
        print('... saving checkoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, name, lr, 
    action_space=None, hidden_dims=256, chkpt_dir="../tmp/sac"):
        super(ActorNetwork, self).__init__()
        self.reparam_noise = 1e-6
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.hidden1 = nn.Linear(*state_dim, hidden_dims)
        self.hidden2 = nn.Linear(hidden_dims, hidden_dims)
        self.mu = nn.Linear(hidden_dims, n_actions)
        self.log_std = nn.Linear(hidden_dims, n_actions)
        
        self.apply(weights_init_)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("mps")
        self.to(self.device)

        if action_space is None:
            self.action_scale = T.tensor(1.)
            self.action_bias = T.tensor(0.)
        else:
            self.action_scale = T.Tensor(
                (action_space.high - action_space.low) / 2.
            ).to(self.device)
            self.action_bias = T.tensor(
                (action_space.high + action_space.low) / 2.
            ).to(self.device)
    
    def forward(self, obs):
        prob = F.relu(self.hidden1(obs))
        prob = F.relu(self.hidden2(prob))
        mu = self.mu(prob)
        log_std = self.log_std(prob)
        log_std = T.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std
    
    def sample_normal(self, state, reparameterize=True):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        probs = Normal(mu, std)
        # reparameterization trick (mean + std * N(0, 1))
        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        action = T.tanh(actions)*self.action_scale.to(self.device) \
            + self.action_bias
        log_probs = probs.log_prob(actions)
        # enforcing action bound
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        mean = T.tanh(mu) * self.action_scale + self.action_bias
        return action, log_probs, mean
    
    def save_checkpoint(self):
        print('... saving checkoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))