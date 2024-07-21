import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt


def plot_learning_curve(scores, x, filename):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, scores)
    plt.title("Running average of prior 100 games")
    plt.savefig(filename)

def update(target, source, tau=None):
    if tau is None: tau = 1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1-tau) + param.data * tau)