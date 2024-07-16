import gymnasium as gym
import numpy as np
from agent import Agent
from noise import OUActionNoise
import matplotlib.pyplot as plt

def plot_learning_curve(scores, x, filename):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.title('Running average of prior 100 games')
    plt.savefig(filename)

N_GAMES = 1000
ENV_NAME = 'LunarLander-v2'
env = gym.make(ENV_NAME, continuous=True)

if __name__ == '__main__':
    agent = Agent(lr1=1e-4, lr2=1e-3, input_dims=env.observation_space.shape,
    tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, 
    n_actions=env.action_space.shape[0])

    filename = f"{ENV_NAME}_alpha_{agent.lr1}_beta_{agent.lr2}_{N_GAMES}_games"
    figure_file = '../plots/' + filename + '.png'
    
    best_score = env.reward_range[0]
    score_history = []

    for i in range(N_GAMES):
        obs, _ = env.reset()
        agent.noise.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, *_ = env.step(action)
            score += reward
            agent.store(obs, action, reward, obs_, done)
            agent.learn()
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode ', i, 'score %.1f' % score,
        'average score %.1f' % avg_score)
    x = [i+1 for i in range(N_GAMES)]
    plot_learning_curve(x, score_history, figure_file)

