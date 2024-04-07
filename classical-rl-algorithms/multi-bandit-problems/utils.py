import numpy as np
import matplotlib.pyplot as plt

def plot_armcount(data, bandit, k):
    x_index = []
    for i in range(k):
        x_index.append(str(i))
    location = np.arange(k)
    (_, eps_history, _, _) = data[bandit]["epsilon"]
    (_, sft_history, _, _) = data[bandit]["softmax"]
    (_, ucb_history, _, _) = data[bandit]["ucb"]
    (_, ts_history, _, _) = data[bandit]["thompson"]
    (fig, ax) = plt.subplots(1,1)
    bar1 = ax.bar(location, eps_history, label="epsilon-greedy", fill=False, edgecolor='green')
    bar2 = ax.bar(location, sft_history, label="softmax", fill=False, edgecolor='red')
    bar3 = ax.bar(location, ucb_history, label="ucb1", fill=False, edgecolor='blue')
    bar3 = ax.bar(location, ts_history, label="thompson", fill=False, edgecolor='orange')
    ax.set_ylabel('Arm Pull History')
    ax.set_title('Number of times arm pulled')
    ax.set_xticks(location)
    ax.set_xticklabels(x_index)
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_regret(data, num_iters, bandit):
    (_, _, eps_regret, _) = data[bandit]["epsilon"]
    # (_, _, sft_regret, _) = data[bandit]["softmax"]
    # (_, _, ucb_regret, _) = data[bandit]["ucb"]
    t = np.arange(num_iters)
    plt.plot(t, eps_regret, color='green', label="epsilon")
    # plt.plot(t, sft_regret, color='red', label="softmax")
    # plt.plot(t, ucb_regret, color='blue', label="ucb1")
    plt.xlabel("Time steps")
    plt.ylabel("Regret")
    plt.title("Regret for "+bandit+" setting")
    plt.legend()
    plt.show()
