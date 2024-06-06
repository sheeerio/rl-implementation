from random import sample

# import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import matplotlib
import torch
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, ob):  # save a transition
        self.buffer[self.index] = ob
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.size + 1) % self.max_size

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.size


class QNet(nn.Module):
    def __init__(self, n_state, n_action):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(n_state, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_action)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# def plot_durations(show_result = False):
#     plt.figure(1)
#     durations_t = torch.tensor(episodes_durations, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0,100,1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#     plt.pause(0.001)


is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_learning_curve(episode, scores, epsilon):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title("episode %s. average_reward: %s" % (episode, np.mean(scores[-10:])))
    plt.plot(scores)
    plt.subplot(132)
    plt.title("epsilon")
    plt.plot(epsilon)
    plt.show()


def plot_playing_curve(episode, scores):
    clear_output(True)
    plt.figure(figsize=(5, 5))
    plt.title("episode %s. average_reward: %s" % (episode, np.mean(scores[-10:])))
    plt.plot(scores)
    plt.show()


def plot_durations(scores, pause):
    plt.ion()
    plt.figure(2)
    plt.clf()

    durations_t = torch.tensor(scores, dtype=torch.float)
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Scores")
    plt.plot(durations_t.numpy())
    # Take 20 episode averages and plot them too
    if len(durations_t) >= 20:
        means = durations_t.unfold(0, 20, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(19), means))
        plt.plot(means.numpy())

    plt.pause(pause)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
