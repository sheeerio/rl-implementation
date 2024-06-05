import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# make the nn
def mlp(sizes, act=nn.Tanh, out_act=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = act if j < len(sizes)-2 else out_act
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)
    
    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()
    
    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        obs = env.reset()
        done = False
        eps_rew = []

        render_this_epoch = True

        while True:
            if render_this_epoch and render:
                env.render()
            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            batch_acts.append(act)
            eps_rew.append(rew)

            if done:
                ep_ret, ep_len = sum(eps_rew), len(eps_rew)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                batch_weights += [ep_ret] * ep_len
                