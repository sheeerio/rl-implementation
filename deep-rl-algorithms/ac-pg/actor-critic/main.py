from actor_critic import Agent
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np


def plot_learning_curve(x, scores, filename):
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = scores[max(0, i - 100) : (i + 1)]
    plt.plot(x, scores)
    plt.title("Running average of prior 100 games")
    plt.savefig(filename)


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(input_dims=[8], n_actions=4)
    n_games = 2000
    fname = f"ActorCritic_lunar_lander_lr{agent.lr}_{n_games}games"
    figfile = "plots/" + fname + ".png"
    scores = []

    for i in range(n_games):
        obs, _ = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, *_ = env.step(action)
            score += reward
            agent.learn(obs, reward, done, obs_)
            obs = obs_
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("episode ", i, "score %.2f" % score, "average score %.2f" % avg_score)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)
