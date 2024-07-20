import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.categorical as Categorical


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, lr=5e-6, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.actions = nn.Linear(fc2_dims, n_actions)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "mps")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.actions(x)
        v = self.v(x)

        return actions, v


class Agent:
    def __init__(self, input_dims, n_actions, gamma=0.99, lr=5e-6):
        self.lr = lr
        self.gamma = gamma
        self.actorcritic = ActorCriticNetwork(input_dims, n_actions)
        self.log_prob = None

    def choose_action(self, obs):
        obs = T.tensor([obs], dtype=T.float).to(self.actorcritic.device)
        actions, _ = self.actorcritic(obs)
        probs = F.softmax(actions, dim=1)
        action_probs = Categorical.Categorical(probs)
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, done, state_):
        self.actorcritic.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.actorcritic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actorcritic.device)
        state_ = T.tensor([state_], dtype=T.float).to(self.actorcritic.device)

        _, V_s = self.actorcritic(state)
        _, V_s_ = self.actorcritic(state_)

        delta = reward + self.gamma * V_s_ * (1 - int(done)) - V_s

        actor_loss = -self.log_prob * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actorcritic.optimizer.step()
