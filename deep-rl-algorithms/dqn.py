# naive dqn but with replay buffer
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from utils import plot_learning_curve


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dim, fc2_dim, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("mps")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class ReplayBuffer(object):
    def __init__(self, input_dims, max_mem_size=100000):
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        batch_index = np.arange(batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(T.device("mps"))
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(T.device("mps"))
        reward_batch = T.tensor(self.reward_memory[batch]).to(T.device("mps"))
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(T.device("mps"))

        action_batch = self.action_memory[batch]

        return (
            state_batch,
            action_batch,
            new_state_batch,
            reward_batch,
            terminal_batch,
            batch_index,
        )


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        batch_size,
        lr,
        input_dims,
        n_actions,
        eps_end=0.01,
        eps_dec=5e-4,
    ):
        self.gamma = gamma
        self.input_dims = input_dims
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]

        self.memory = ReplayBuffer(input_dims=self.input_dims, n_actions=n_actions)
        self.Q_eval = DeepQNetwork(
            self.lr,
            n_actions=n_actions,
            input_dims=self.input_dims,
            fc1_dim=256,
            fc2_dim=256,
        )

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor(obs, dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        (
            state_batch,
            action_batch,
            new_state_batch,
            reward_batch,
            terminal_batch,
            batch_index,
        ) = self.memory.sample_buffer(self.batch_size)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=[8],
        lr=3e-3,
    )
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        obs, _ = env.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, *_ = env.step(action)
            score += reward
            agent.memory.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(
            "episode ",
            i,
            "score %.2f" % score,
            "average score %.2f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )

    x = [i + 1 for i in range(n_games)]
    filename = "lunar_lander.png"
    plot_learning_curve(x, scores, eps_history, filename)
