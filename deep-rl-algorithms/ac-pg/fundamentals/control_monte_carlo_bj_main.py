import gymnasium as gym
import matplotlib.pyplot as plt
from control_monte_carlo_bj import Agent

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    agent = Agent(eps=0.001)
    n_eps = 200000
    win_lose_draw = {-1:0, 0:0, 1:0}
    win_rates = []
    for i in range(n_eps):
        if i > 0 and i % 1000 == 0:
            pct = win_lose_draw[1] / i
            win_rates.append(pct)
        if i % 50000 == 0:
            rates = win_rates[-1] if win_rates else 0.0
            print('running episode ', i, 'win rate %.3f' % rates)
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, *_ = env.step(action)
            agent.memory.append((state, action, reward))
            state = state_
        agent.update_q()
        win_lose_draw[reward] += 1
    plt.plot(win_rates)
    plt.show()