#!/usr/bin/env python3
import os
import time
import torch as T

from buffer import ReplayBuffer
import utils

import gymnasium as gym
import hydra


class Workspace(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make('Hopper-v4')

        utils.set_seed_everywhere(1)
        self.device = T.device("mps")

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(1e6),
                                          self.device)

        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for _ in range(20000):
            obs, _ = self.env.reset()
            # self.agent.reset()
            done = False
            episode_reward = 0
            while not done:
                with utils.set_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward

            average_episode_reward += episode_reward
        average_episode_reward /= 10

    def run(self):
        episode, episode_reward, done = 0, 0, True
        while self.step < 1000000:
            if done:
                obs, _ = self.env.reset()
                # self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

            # sample action for data collection
            if self.step < 5000:
                action = self.env.action_space.sample()
            else:
                # with utils.set_mode((cfg.agent.critic_cfg, cfg.agent.actor_cfg)):
                action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= 5000:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, *_ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config', config_name='sac')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()