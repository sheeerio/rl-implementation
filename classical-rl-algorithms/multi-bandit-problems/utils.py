import numpy as np
import matplotlib.pyplot as plt

def plot_armcount(data, bandit, k):
    x_index = []
    for i in range(k):
        x_index.append(str(i))
    location = np.arange(k)
    (_, eps_history, _, _) = data[bandit]["epsilon"]
    (_, sft_history, _, _) = data[bandit]["softmax"]
    (fig, ax) = plt.subplots(1,1)
    bar1 = ax.bar(location, eps_history, label="epsilon-greedy", fill=False, edgecolor='green')
    bar2 = ax.bar(location, sft_history, label="softmax", fill=False, edgecolor='red')
    ax.set_ylabel('Arm Pull History')
    ax.set_title('Number of times arm pulled')
    ax.set_xticks(location)
    ax.set_xticklabels(x_index)
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_regret(data, num_iters, bandit, player):
    (_, _, regret, _) = data[bandit][player]
    t = np.arange(num_iters)
    plt.plot(t, regret, color='green', label=player)
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()
