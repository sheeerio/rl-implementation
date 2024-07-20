import numpy as np


class ReplayBuffer:
    def __init__(self, size, input_dims, n_actions):
        self.mem_size = size
        self.mem_cntr = 0
        self.states = np.zeros((self.mem_size, *input_dims))
        self.actions = np.zeros((self.mem_size, n_actions))
        self.states_ = np.zeros((self.mem_size, *input_dims))
        self.terminals = np.zeros(self.mem_size)
        self.rewards = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, done, state_):
        idx = self.mem_cntr % self.mem_size
        self.states[idx] = state
        self.states_[idx] = state_
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminals[idx] = done
        self.mem_cntr += 1

    def sample_transition(self, batch_size):
        current_size = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(current_size, batch_size, replace=False)
        states = self.states[batch]
        states_ = self.states_[batch]
        rewards = self.rewards[batch]
        actions = self.actions[batch]
        dones = self.terminals[batch]

        return states, states_, rewards, dones, actions
