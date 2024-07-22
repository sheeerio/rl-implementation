import numpy as np
import torch as T

class ReplayBuffer:
    """Buffer to store experiences.

    Attributes:
        capacity (int): The maximum transitions the buffer can hold.
        device (torch.device): Device where tensors will be moved.
        obses (np.ndarray): Array to store observations.
        obses_ (np.ndarray): Array to store transitionary observations/
        actions (np.ndarray): Array to store actions taken by some policy/
        rewards (np.ndarray): Array to store rewards received after taking action.
        not_dones (np.ndarray): Array to store the negation of the `done` flag. 
            `1` indicates the episode does not terminate, `0` indicates the episode terminates.
        not_dones_no_max (np.ndarray): Array to store the negation of the `done` flag.
            `1` indicates the episode does not terminate or ended due to reaching maximum steps,
            `0` indicates the episode terminates naturally.
        idx (int): The current index in the buffer where the next transition will be stored.
        last_save (int): The index of the last saved transition.
        full (bool): Indicator if the buffer is full, meaning it has stored `capacity` transitions.
    """
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.obses_ = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
    
    def __len__(self):
        """
        Returns the number of transitions stored in the buffer.
        """
        return self.capacity if self.full else self.idx
    
    def add(self, obs, action, reward, obs_, done, done_no_max):
        """
        Adds a new transition to the buffer.
        """
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.obses_[self.idx], obs_)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        # true when latest transtion stored is at idx = capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.
        """
        idxs = np.random.randint(0, self.__len__(), size=batch_size)
        
        obses = T.as_tensor(self.obses[idxs], device=self.device).float()
        actions = T.as_tensor(self.actions[idxs], device=self.device)
        rewards = T.as_tensor(self.rewards[idxs], device=self.device)
        obses_ = T.as_tensor(self.obses_[idxs], device=self.device).float()
        not_dones = T.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = T.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, obses_, not_dones, not_dones_no_max