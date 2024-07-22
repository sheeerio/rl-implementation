import numpy as np
import random

class SumTree:
    """
    Useful for computing the cumsum of all transition priorities. \
    Each node of a SumTree has the following attributes (edified):
        Node (object):
            self.parent (node): contains the sum of all descendent nodes
            self.left (node): left child of current node
            self.right (node): right child of current node
            self.value (int): contains the transition priority of the \
                current transition 'node' + sum of all descendent nodes.
        
    Attributes:
        nodes (list[int]): Array representation of the SumTree nodes.
        data (list[int]): Array representation for the `node.data`.
        size (int): The maximum transitions the SumTree can hold.
        curr_size (int): The total number of transitions currently in the SumTree.
        count (int): 
    """
    def __init__(self, size):
        
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.curr_size = 0
        self.count = 0
    
    @property
    def total(self):
        return self.nodes[0]
    
    def update(self, data_idx, value):
        """
        Trickles up the sum change from leaf up to the root.

        Args:
            data_idx (int): The index corresponding to the data in `self.data`
            value (int): The recent value added to the heap (`self.data[self.count]`).
        """
        idx = data_idx + self.curr_size - 1 # get node index corresponding to data_idx
        change = value - self.nodes[idx]

        self.nodes[idx] = value
        # trickle up the change to the parents
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.curr_size = min(self.curr_size + 1, self.size)

    def get(self, cumsum):
        assert cumsum <= self.total

        # start search from root
        idx = 0
        while 2 * idx + 1 <= len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum -= self.nodes[left]
        # node index to data_idx
        data_idx = idx - self.curr_size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]
        
    def __len__(self):
        return self.curr_size

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"

class PrioritizedReplayBuffer(object):
    """
    Class for prioritized experience replay (PER) buffer.

    Attributes:
        tree (SumTree): The sum-tree data-structure holding the transitions.
        eps (float): Minimum possible priority for any transition, to avoid zero probabilities.
        alpha (float): Determines how much prioritization is used.
            `1` corresponds to uniform priorities across all transitions.
        beta (float): Determines the amount of importance sampling weight correction.
            `1` corresponds to fully compensating for non-uniform probabilities.
        max_priority (float): Priority set for newly added transitions. \
            This is never smaller than the maximum priority in the buffer.
        
    """
    def __init__(self, state_shape, action_shape, max_size, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(max_size)

        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps

        obs_dtype = np.float32 if len(state_shape) == 1 else np.uint8

        self.obses = np.zeros((max_size, *state_shape), dtype=obs_dtype)
        self.obses_ = np.zeros((max_size, *state_shape), dtype=obs_dtype)
        self.actions = np.zeros((max_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.not_dones = np.zeros((max_size, 1), dtype=np.float32)
        self.not_dones_no_max = np.zeros((max_size, 1), dtype=np.float32)

        self.idx = 0
        self.curr_size = 0
        self.capacity = max_size

    def __len__(self):
        return self.curr_size

    def add(self, obs, action, reward, obs_, done, done_no_max):
        self.tree.add(self.max_priority, self.count)

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.obses_[self.idx], obs_)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.curr_size = min(self.curr_size + 1, self.size)
    
    def sample(self, batch_size):
        assert self.batch_size >= self.curr_size, "buffer contains less samples than sample size of `batch_size`"

        sample_idxs, tree_idxs = [], []
        priorities = np.empty((batch_size, 1), dtype=np.float32)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i+1)

            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            sample_idxs.append(sample_idx)
            tree_idxs.append(tree_idx)
        
        probs = priorities / self.tree.total

        weights = (self.curr_size * probs) ** -self.beta
        weights /= weights.max()

        return self.obses[sample_idxs], self.obses_[sample_idxs], self.rewards[sample_idxs], \
            self.actions[sample_idxs], self.not_dones[sample_idxs], self.not_dones_no_max[sample_idxs], \
                weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):

        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)

            self.max_priority = max(self.max_priority, priority)