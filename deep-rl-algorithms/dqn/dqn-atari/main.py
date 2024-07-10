import argparse, os
import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import agents as Agents
from utils import plot_learning_curve, make_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q Learning")
    parser.add_argument("-n_games", type=int, default=1, help="number of games to play")
    parser.add_argument(
        "-lr", type=float, default=1e-4, help="learning rate for optimizer"
    )
    parser.add_argument(
        "-eps_min",
        type=float,
        default=0.1,
        help="minimum value for epsilon in epsilon-greedy action selection",
    )
    parser.add_argument(
        "-gamma", type=float, default=0.99, help="discount factor for bellman update"
    )
    parser.add_argument(
        "-eps_dec",
        type=float,
        default=1e-5,
        help="linear factor for decreasing epsilon",
    )
    parser.add_argument(
        "-eps",
        type=float,
        default=1.0,
        help="starting value for epsilon in epsilon-greedy action selection",
    )
    parser.add_argument(
        "-max_mem",
        type=int,
        default=18000,
        help="maximum size for memory replay buffer",
    )
    parser.add_argument(
        "-repeat", type=int, default=4, help="number of frames to stack for environment"
    )
    parser.add_argument(
        "-bs", type=int, default=32, help="batch size for replay memory sampling"
    )
    parser.add_argument(
        "-replace", type=int, default=1000, help="interval for replacing target network"
    )
    parser.add_argument(
        "-env",
        type=str,
        default="PongNoFrameskip-v4",
        help="atari environment.\nPongNoFrameskip-v4\n \
        BreakoutNoFrameskip-v4\n SpaceInvadersNoFrameskip-v4\n \
        EnduroNoFrameskip-v4\n AtlantisNoFrameskip-v4",
    )
    parser.add_argument(
        "-device", type=str, default="mps", help="GPU: mps or cuda:0 or cuda:1"
    )
    parser.add_argument(
        "-load_checkpoint", type=bool, default=False, help="load model checkpoint"
    )
    parser.add_argument(
        "-path", type=str, default="models/", help="path for model saving/loading"
    )
    parser.add_argument(
        "-algo",
        type=str,
        default="DQNAgent",
        help="DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent",
    )
    parser.add_argument(
        "-clip_rewards", type=bool, default=False, help="clip rewards to range -1 to 1"
    )
    parser.add_argument(
        "-no_ops", type=int, default=0, help="max number of no ops for testing"
    )
    parser.add_argument(
        "-fire_first",
        type=bool,
        default=False,
        help="set first action of episode to fire",
    )
    args = parser.parse_args()

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.load_checkpoint:
        env = make_env(
            env_name=args.env,
            repeat=args.repeat,
            clip_rewards=args.clip_rewards,
            no_ops=args.no_ops,
            fire_first=args.fire_first,
            render_mode="human",
        )
    else:
        env = make_env(
            env_name=args.env,
            repeat=args.repeat,
            clip_rewards=args.clip_rewards,
            no_ops=args.no_ops,
            fire_first=args.fire_first,
        )

    agent_ = getattr(Agents, args.algo)
    best_score = -np.inf

    agent = agent_(
        gamma=args.gamma,
        epsilon=args.eps,
        lr=args.lr,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
        memory_size=args.max_mem,
        eps_min=args.eps_min,
        batch_size=args.bs,
        replace=args.replace,
        eps_dec=args.eps_dec,
        chkpt_dir=args.path,
        algo=args.algo,
        env_name=args.env,
        device=args.device,
    )

    if args.load_checkpoint:
        agent.load_models()

    fname = f"{args.algo}_{args.env}_lr{args.lr}_{args.n_games}games"
    figure_file = "plots/" + fname + ".png"

    scores_file = fname + "_scores.npy"

    scores, eps_history = [], []
    n_steps = 0
    steps_array = []
    for i in range(args.n_games):
        done = False
        obs, _ = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, *_ = env.step(action)
            score += reward

            if not args.load_checkpoint:
                agent.store_transition(obs, action, reward, int(done), obs_)
                agent.learn()
            obs = obs_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print(
            "episode: ",
            i,
            "score: ",
            score,
            " average score %.1f" % avg_score,
            "best score %.2f" % best_score,
            "epsilon %.2f" % agent.epsilon,
            "steps",
            n_steps,
        )

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if args.load_checkpoint and n_steps >= 18000:
            break

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
    np.save(scores_file, np.array(scores))
