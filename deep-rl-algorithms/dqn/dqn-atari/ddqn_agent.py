import numpy as np
import torch as T
from dqn import DQN, DuelingDQN
from utils import ReplayBuffer


class DuelingDQNAgent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        n_actions,
        input_dims,
        memory_size,
        batch_size,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        algo=None,
        env_name=None,
        chkpt_dir="../../tmp/dueling_dqn",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size=memory_size, state_dim=input_dims)

        self.q_val = DuelingDQN(
            self.input_dims,
            self.env_name + "_" + self.algo + "_q_eval",
            self.n_actions,
            self.lr,
            self.chkpt_dir,
        )
        self.q_next = DuelingDQN(
            self.input_dims,
            self.env_name + "_" + self.algo + "_q_next",
            self.n_actions,
            self.lr,
            self.chkpt_dir,
        )

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor([obs], dtype=T.float).to(self.q_val.device)
            _, advantage = self.q_val(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, done, state_):
        self.memory.store(state, state_, action, reward, done)

    def sample_transition(self):
        state, state_, action, reward, done = self.memory.sample(self.batch_size)
        states = T.tensor(state).to(self.q_val.device)
        states_ = T.tensor(state_).to(self.q_val.device)
        actions = T.tensor(action).to(self.q_val.device)
        rewards = T.tensor(reward).to(self.q_val.device)
        dones = T.tensor(done).to(self.q_val.device)

        return states, actions, rewards, dones, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_val.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_models(self):
        self.q_val.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_val.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_val.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, dones, states_ = self.sample_transition()

        indices = np.arange(self.batch_size)

        value, advantage = self.q_val.forward(states)
        q_next_val, q_next_adv = self.q_next.forward(states_)

        q_val = T.add(value, (advantage - advantage.mean(dim=1, keepdims=True)))[
            indices, actions
        ]
        q_next = T.add(
            q_next_val, (q_next_adv - q_next_adv.mean(dim=1, keepdims=True))
        ).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next
        loss = self.q_val.loss(q_target, q_val).to(self.q_val.device)
        loss.backward()

        self.q_val.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DuelingDoubleDQNAgent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        n_actions,
        input_dims,
        memory_size,
        batch_size,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        algo=None,
        env_name=None,
        chkpt_dir="../../tmp/dueling_double_dqn",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size=memory_size, state_dim=input_dims)

        self.q_val = DuelingDQN(
            self.input_dims,
            self.env_name + "_" + self.algo + "_q_eval",
            self.n_actions,
            self.lr,
            self.chkpt_dir,
        )
        self.q_next = DuelingDQN(
            self.input_dims,
            self.env_name + "_" + self.algo + "_q_next",
            self.n_actions,
            self.lr,
            self.chkpt_dir,
        )

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor([obs], dtype=T.float).to(self.q_val.device)
            _, advantage = self.q_val(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, done, state_):
        self.memory.store(state, state_, action, reward, done)

    def sample_transition(self):
        state, state_, action, reward, done = self.memory.sample(self.batch_size)
        states = T.tensor(state).to(self.q_val.device)
        states_ = T.tensor(state_).to(self.q_val.device)
        actions = T.tensor(action).to(self.q_val.device)
        rewards = T.tensor(reward).to(self.q_val.device)
        dones = T.tensor(done).to(self.q_val.device)

        return states, actions, rewards, dones, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_val.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_models(self):
        self.q_val.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_val.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_val.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, dones, states_ = self.sample_transition()

        indices = np.arange(self.batch_size)

        value, advantage = self.q_val.forward(states)
        q_next_val, q_next_adv = self.q_next.forward(states_)
        q_pred_val, q_pred_adv = self.q_val.forward(states_)

        q_val = T.add(value, (advantage - advantage.mean(dim=1, keepdims=True)))[
            indices, actions
        ]
        q_next = T.add(q_next_val, (q_next_adv - q_next_adv.mean(dim=1, keepdims=True)))
        q_pred = T.add(q_pred_val, (q_pred_adv - q_pred_adv.mean(dim=1, keepdims=True)))
        max_actions = T.argmax(q_pred, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_val.loss(q_target, q_val).to(self.q_val.device)
        loss.backward()

        self.q_val.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()


class DDQNAgent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        n_actions,
        input_dims,
        memory_size,
        batch_size,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        algo=None,
        env_name=None,
        chkpt_dir="../../tmp/ddqn",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size=memory_size, state_dim=input_dims)

        self.q_val = DQN(
            self.input_dims,
            self.env_name + "_" + self.algo + "_q_eval",
            self.n_actions,
            self.lr,
            self.chkpt_dir,
        )
        self.q_next = DQN(
            self.input_dims,
            self.env_name + "_" + self.algo + "_q_next",
            self.n_actions,
            self.lr,
            self.chkpt_dir,
        )

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            state = T.tensor([obs], dtype=T.float).to(self.q_val.device)
            actions = self.q_val(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, done, state_):
        self.memory.store(state, state_, action, reward, done)

    def sample_transition(self):
        state, state_, action, reward, done = self.memory.sample(self.batch_size)
        states = T.tensor(state).to(self.q_val.device)
        states_ = T.tensor(state_).to(self.q_val.device)
        actions = T.tensor(action).to(self.q_val.device)
        rewards = T.tensor(reward).to(self.q_val.device)
        dones = T.tensor(done).to(self.q_val.device)

        return states, actions, rewards, dones, states_

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_val.state_dict())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def save_models(self):
        self.q_val.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_val.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_val.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, dones, states_ = self.sample_transition()

        indices = np.arange(self.batch_size)

        q_val = self.q_val.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_pred = self.q_val.forward(states_)

        max_actions = T.argmax(q_pred, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_val.loss(q_target, q_val).to(self.q_val.device)
        loss.backward()

        self.q_val.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()
