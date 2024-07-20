import os
import numpy as np

import torch as T
import torch.nn.functional as F

from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        lr1,
        lr2,
        tau,
        env,
        input_dims,
        n_actions,
        gamma=0.99,
        max_size=1000000,
        fc1_dims=400,
        fc2_dims=300,
        batch_size=100,
        update_actor_interval=2,
        warmup=1000,
        noise=0.1,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.update_actor_interval = update_actor_interval
        self.learn_step_cntr = 0
        self.warmup = warmup
        self.action_step = 0
        self.lr1 = lr1
        self.lr2 = lr2
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            input_dims, n_actions, lr1, fc1_dims, fc2_dims, name="actor"
        )
        self.critic1 = CriticNetwork(
            input_dims, n_actions, lr2, fc1_dims, fc2_dims, name="critic1"
        )
        self.critic2 = CriticNetwork(
            input_dims, n_actions, lr2, fc1_dims, fc2_dims, name="critic2"
        )

        self.actor_ = ActorNetwork(
            input_dims, n_actions, lr1, fc1_dims, fc2_dims, name="target_actor"
        )
        self.critic1_ = CriticNetwork(
            input_dims, n_actions, lr2, fc1_dims, fc2_dims, name="target_critic1"
        )
        self.critic2_ = CriticNetwork(
            input_dims, n_actions, lr2, fc1_dims, fc2_dims, name="target_critic2"
        )

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        if self.action_step < self.warmup:
            action = T.normal(0, self.noise, size=(self.n_actions,)).to(
                self.actor.device
            )
        else:
            obs = T.tensor(obs, dtype=T.float).to(self.actor.device)
            action = self.actor(obs) + T.normal(
                0, self.noise, size=(self.n_actions,)
            ).to(self.actor.device)
        action = T.clamp(action, self.min_action[0], self.max_action[0])
        self.action_step += 1

        return action.cpu().detach().numpy()

    def store(self, state, action, reward, done, state_):
        self.memory.store_transition(state, action, reward, done, state_)

    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic1_.save_checkpoint()
        self.critic2.save_checkpoint()
        self.critic2_.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic1_.load_checkpoint()
        self.critic2.load_checkpoint()
        self.critic2_.load_checkpoint()

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

        actions_ = self.actor_(states_) + T.clamp(
            T.tensor(np.random.normal(scale=0.2), device=self.actor.device), -0.5, 0.5
        )
        actions_ = T.clamp(actions_, self.min_action[0], self.max_action[0])
        # might break if min and max of action space are not equal

        Q1_ = self.critic1_(states_, actions_).view(-1)
        Q2_ = self.critic2_(states_, actions_).view(-1)
        Q1_[dones] = 0.0
        Q2_[dones] = 0.0

        Q1 = self.critic1(states, actions)
        Q2 = self.critic2(states, actions)

        Q_ = T.min(Q1_, Q2_)

        target = (rewards + self.gamma * Q_).view(self.batch_size, 1)

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        critic1_loss = F.mse_loss(target, Q1)
        critic2_loss = F.mse_loss(target, Q2)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()

        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_loss = -T.mean(self.critic1(states, self.actor(states)))
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        critic1_params = dict(self.critic1.named_parameters())
        critic2_params = dict(self.critic2.named_parameters())
        _actor_params = dict(self.actor_.named_parameters())
        _critic1_params = dict(self.critic1_.named_parameters())
        _critic2_params = dict(self.critic2_.named_parameters())

        for name1, name2, name3 in zip(critic1_params, critic2_params, actor_params):
            critic1_params[name1] = (
                tau * critic1_params[name1].clone()
                + (1 - tau) * _critic1_params[name1].clone()
            )
            critic2_params[name2] = (
                tau * critic2_params[name2].clone()
                + (1 - tau) * _critic2_params[name2].clone()
            )
            actor_params[name3] = (
                tau * actor_params[name3].clone()
                + (1 - tau) * _actor_params[name3].clone()
            )

        self.actor_.load_state_dict(actor_params)
        self.critic1_.load_state_dict(critic1_params)
        self.critic2_.load_state_dict(critic2_params)
