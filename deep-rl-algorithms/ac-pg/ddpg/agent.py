import torch as T
import torch.nn.functional as F
import numpy as np

from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from noise import OUActionNoise


class Agent:
    def __init__(
        self,
        lr1,
        lr2,
        tau,
        input_dims,
        n_actions,
        gamma=0.99,
        max_size=1000000,
        fc1_dims=400,
        fc2_dims=300,
        batch_size=64,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr1 = lr1
        self.lr2 = lr2

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(
            lr1, input_dims, fc1_dims, fc2_dims, n_actions, name="actor"
        )
        self.critic = CriticNetwork(
            lr2, input_dims, fc1_dims, fc2_dims, n_actions, name="critic"
        )
        self.target_actor = ActorNetwork(
            lr1, input_dims, fc1_dims, fc2_dims, n_actions, name="target_actor"
        )
        self.target_critic = CriticNetwork(
            lr2, input_dims, fc1_dims, fc2_dims, n_actions, name="target_critic"
        )

        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        self.actor.eval()
        obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
        mu = self.actor(obs).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def store(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, done, state_)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, states_, rewards, dones, actions = self.memory.sample_transition(
            self.batch_size
        )
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.actor.device)

        target_actions = self.target_actor(states_)
        Q_ = self.target_critic(states_, target_actions).view(self.batch_size)
        Q = self.critic(states, actions)

        Q_[dones] = 0.0
        Q_ = Q_  # .view(-1)

        target = (rewards + self.gamma * Q_).view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, Q)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        for name in critic_params:
            critic_params[name] = (
                tau * critic_params[name].clone()
                + (1 - tau) * target_critic_params[name].clone()
            )
        for name in actor_params:
            actor_params[name] = (
                tau * actor_params[name].clone()
                + (1 - tau) * target_actor_params[name].clone()
            )

        self.target_critic.load_state_dict(critic_params)
        self.target_actor.load_state_dict(actor_params)
