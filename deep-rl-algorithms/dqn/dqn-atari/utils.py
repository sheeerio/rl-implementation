import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", colors="C0")
    ax.tick_params(axis="y", colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 100) : (t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")

    plt.savefig(filename)


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(
        self, env=None, repeat=4, clip_rewards=False, no_ops=0, fire_first=False
    ):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.shape = self.env.observation_space.low.shape
        self.buffer = np.zeros_like((2, *self.shape))
        self.clip_rewards = clip_rewards
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        total_rew = 0.0
        done = False
        for i in range(self.repeat):
            obs, rew, done, *lame = self.env.step(action)
            if self.clip_rewards:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            total_rew += rew
            idx = i % 2
            self.buffer[idx] = obs
            if done:
                break
        # find the max frame
        max_frame = np.maximum(self.buffer[0], self.buffer[1])

        return max_frame, total_rew, done, *lame

    def reset(self, **kwargs):
        obs, lame = self.env.reset(**kwargs)
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, *_ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
            obs, *_ = self.env.step(1)

        self.buffer = np.zeros((2, *self.shape))
        self.buffer[0] = obs

        return obs, lame


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env=None, *shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=self.shape, dtype=np.float32
        )

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(
            new_frame, self.shape[1:], interpolation=cv2.INTER_AREA
        )
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, stack_size):
        super(StackFrames, self).__init__(env)
        self.stack_size = stack_size
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(stack_size, axis=0),
            env.observation_space.high.repeat(stack_size, axis=0),
            dtype=np.float32,
        )
        self.frame_stack = deque(maxlen=stack_size)

    def reset(self, **kwargs):
        self.frame_stack.clear()
        obs, lame = self.env.reset(**kwargs)
        for _ in range(self.stack_size):
            self.frame_stack.append(obs)

        return (
            np.array(self.frame_stack).reshape(self.observation_space.low.shape),
            lame,
        )

    def observation(self, input):
        self.frame_stack.append(input)

        return np.array(self.frame_stack).reshape(self.observation_space.low.shape)


def make_env(
    env_name,
    new_shape=(84, 84, 1),
    repeat=4,
    clip_rewards=False,
    no_ops=0,
    fire_first=False,
):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(env, *new_shape)
    env = StackFrames(env, repeat)

    return env


class ReplayBuffer:
    def __init__(self, mem_size, state_dim):
        self.max_mem = mem_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.max_mem, *state_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.max_mem, *state_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.max_mem, dtype=np.int64)
        self.reward_mem = np.zeros(self.max_mem, dtype=np.float32)
        self.terminal_mem = np.zeros(self.max_mem, dtype=np.uint8)

    def store(self, state, state_, action, reward, done):
        idx = self.mem_cntr % self.max_mem
        self.state_mem[idx] = state
        self.new_state_mem[idx] = state_
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.terminal_mem[idx] = done
        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.max_mem, self.mem_cntr)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        states_ = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]

        return states, states_, actions, rewards, dones
