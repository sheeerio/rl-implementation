import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

import hydra

import utils
from agent import Agent

class SAC(Agent):
    """SAC agent"""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_freq, critic_lr,
                 critic_betas, critic_tau, critic_target_update_freq,
                 batch_size, learnable_temperature):
        super().__init__()

        self.action_range = action_range
        self.device = T.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_ = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = T.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),
        lr=actor_lr,betas=actor_betas)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(),
        lr=critic_lr,betas=critic_betas)
        self.log_alpha_optimizer = T.optim.Adam([self.log_alpha],
        lr=alpha_lr,betas=alpha_betas)

        self.train()
        self.critic_.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def act(self, obs, sample=False):
        obs = T.FloatTensor(obs).to(self.device).unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])
    
    def update_critic(self, obs, action, reward, obs_, not_done):
        dist = self.actor(obs_)
        action_ = dist.rsample()
        log_prob = dist.log_prob(action_).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_(obs_, action_)
        target_V = T.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) \
            + F.mse_loss(current_Q2, target_Q)
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = T.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * 
            (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
    
    def update(self, buffer, step):
        obs, action, reward, obs_, not_done, not_done_no_max = buffer.sample(
            self.batch_size)
        
        self.update_critic(obs, action, reward, obs_, not_done_no_max)

        if step % self.actor_update_freq == 0:
            self.update_actor_alpha(obs)
        
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_, self.critic_tau)