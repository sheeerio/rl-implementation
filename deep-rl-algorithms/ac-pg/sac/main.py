import argparse
import torch as T
import gymnasium as gym
import numpy as np

import wandb

from agent import SAC
from utils import plot_learning_curve

wandb.login()

parser = argparse.ArgumentParser(description="Soft Actor Critic")
parser.add_argument('-env_name', default='Hopper-v4', 
                    help='Mujoco Gym environment (default: Hopper-v4)')
parser.add_argument('-eval', type=bool, default=False,
                    help='Evaluates a policy every 10 episode (default: False)')
parser.add_argument('-gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('-reward_scale', type=float, default=5, metavar='G',
                    help='reward scale (default: 5)')
parser.add_argument('-tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('-lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('-alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('-automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('-seed', type=int, default=420, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('-batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('-num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('-hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('-start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('-target_update_interval', type=int, default=1000, metavar='N',
                    help='Value target (hard) update per no. of updates per step (default: 1000)')
parser.add_argument('-max_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('-cuda', action="store_true",
                    help='run on CUDA (default: False)')

N_GAMES = 500

args = parser.parse_args()

run = wandb.init(
    # Set the project where this run will be logged
    project="soft-actor-critic",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "epochs": N_GAMES,
    },
)
wandb.config.update(args)

env = gym.make(args.env_name)
env.action_space.seed(args.seed)
T.manual_seed(args.seed)
np.random.seed(args.seed)

agent = SAC(env.observation_space.shape, env.action_space, args)

filename = f"{args.env_name}_{N_GAMES}_{agent.scale}_scale.png"
figure_file = "../plots/" + filename

score_history, critic_losses, actor_losses = [], [], []
total_timesteps = 0

if __name__ == '__main__':
    if args.eval:
        agent.load_models()
        env = gym.make(args.env_name, render_mode="human")

    for i in range(N_GAMES):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs, args.eval)
            obs_, reward, done, *_ = env.step(action)
            score += reward
            obs = obs_
            total_timesteps += 1
            if not args.eval:
                agent.memory.store_transition(obs, action, reward, done, obs_)
                agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if total_timesteps > args.num_steps:
            break
        if i % 5 == 0:
            print("episode ", i, "score %.1f" % score, "avg score %.1f" % avg_score, "total timesteps", total_timesteps)
            wandb.log({"avg_score": avg_score})
    
    if not args.eval:
        x = [i + 1 for i in range(N_GAMES)]
        plot_learning_curve(x, score_history, figure_file)
    