import numpy as np


class Agent:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.V = {}
        self.sum_space = [i for i in range(4, 22)]
        self.dealer_show_card_space = [i + 1 for i in range(10)]
        self.ace_space = [False, True]
        self.action_space = [0, 1]  # stick or hit

        self.state_space = []
        self.returns = {}
        self.states_visited = {}  # first visit or not
        self.memory = []

        self.initialize_value()

    def initialize_value(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    state = (total, card, ace)
                    self.V[state] = 0
                    self.returns[state] = []
                    self.states_visited[state] = 0
                    self.state_space.append(state)

    def policy(self, state):
        total, *_ = state
        action = 0 if total >= 20 else 1
        return action

    def update_v(self):
        for idt, (state, _) in enumerate(self.memory):
            G = 0
            if self.states_visited[state] == 0:
                self.states_visited[state] = 1
                discount = 1
                for _, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        for state in self.state_space:
            self.states_visited[state] = 0

        self.memory = []
