import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym
from utils import plot_learning_curve
import os


class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.max_mem = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.max_mem, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.max_mem, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.max_mem

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.max_mem, self.mem_cntr)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        batch_indices = np.arange(batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch], dtype=T.float).to(
            T.device("mps")
        )
        new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float).to(
            T.device("mps")
        )
        reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float).to(
            T.device("mps")
        )
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.float).to(
            T.device("mps")
        )
        action_batch = self.action_memory[batch]

        return (
            state_batch,
            new_state_batch,
            reward_batch,
            terminal_batch,
            action_batch,
            batch_indices,
        )


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        n_actions,
        name,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="tmp/sac",
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(T.device("mps"))

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        fc1_dims=256,
        fc2_dims=256,
        name="value",
        chkpt_dir="tmp/sac",
    ):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(T.device("mps"))

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
        v = self.v(value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        max_action,
        input_dims,
        fc1_dims=256,
        fc2_dims=256,
        name="actor",
        n_actions=2,
        chkpt_dir="tmp/sac",
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "_sac")
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(T.device("mps"))

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = F.relu(self.fc2(prob))

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(T.device("mps"))
        log_probs = probabilities.log_prob(actions)

        # Debugging: Print the shape of log_probs
        # print(f"log_probs shape before summing: {log_probs.shape}")
        # print(f"action shape before summing: {action.shape}")

        # Ensure log_probs has the correct dimensions
        log_probs = log_probs.view(log_probs.size(0), -1)  # Ensure it's at least 2D
        log_probs = log_probs.sum(1, keepdim=True)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(
        self,
        alpha=3e-4,
        beta=3e-4,
        input_dims=8,
        tau=5e-3,
        env=None,
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            alpha,
            input_dims=input_dims,
            n_actions=n_actions,
            name="actor",
            max_action=env.action_space.high,
        )
        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions=n_actions, name="critic_2"
        )
        self.value = ValueNetwork(beta, input_dims, name="value")
        self.target_value = ValueNetwork(beta, input_dims, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        state = T.Tensor(obs).to(T.device("mps"))
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        # print(actions.shape)
        return actions.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done, batch_index = self.memory.sample_buffer(
            self.batch_size
        )

        value = self.value(state).view(-1)
        value_ = self.target_value(new_state).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)[batch_index, action]
        q2_new_policy = self.critic_2.forward(state, actions)[batch_index, action]
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value_optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action)[batch_index, action]
        q2_old_policy = self.critic_2.forward(state, action)[batch_index, action]
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4")
    agent = Agent(
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
    )
    n_games = 250
    filename = "inverted_pendulum.png"

    figure_file = "plots/" + filename
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode="human")

    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            # print(action.shape)
            obs_, reward, done, *_ = env.step(action)
            score += reward
            agent.remember(obs, action, reward, obs_, done)
            if not load_checkpoint:
                agent.learn()
            obs = obs_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print("episode ", i, "score %.1f" % score, "avg_score %.1f" % avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, None, figure_file)
