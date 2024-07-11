import gymnasium as gym
from pred_monte_carlo_bj import Agent

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    agent = Agent()
    n_eps = 500000
    for i in range(n_eps):
        if i % 50000 == 0:
            print("starting episode ", i)
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.policy(obs)
            obs_, reward, done, *_ = env.step(action)
            agent.memory.append((obs, reward))
            obs = obs_
        agent.update_v()
    print(agent.V[(21, 3, True)])
    print(agent.V[(4, 1, False)])
