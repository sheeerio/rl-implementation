import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from reinforce_ import Agent


def plot_learning_curve(scores, x, filename):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, scores)
    plt.title("Running average of prior 100 games")
    plt.savefig(filename)


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(lr=0.0005, gamma=0.99, input_dims=[8], n_actions=4)
    n_games = 3000
    fname = f"REINFORCE_lunar_lunar_lr{agent.lr}_{n_games}games"
    figure_file = "plots/" + fname + ".png"
    scores = []

    for i in range(n_games):
        obs, _ = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, *_ = env.step(action)
            score += reward
            agent.store_rewards(reward)
            obs = obs_
        scores.append(score)
        agent.learn()

        avg_score = np.mean(scores[-100:])
        print("episode ", i, "score %.2f" % score, "average score %.2f" % avg_score)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)
