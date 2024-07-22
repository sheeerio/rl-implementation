import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class set_mode(object):
    def __init__(self, *models, train=False):
        self.models = models
        self.train = train
    
    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(self.train)
    
    def __exit__(self):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def set_seed_everywhere(seed):
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(source, target, tau):
    """
    Soft update target network parameters
    """
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

def weight_init(m):
    """
    Custom weight init for Conv2D and linear layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

def mlp(input_dim:int, hidden_dim:int, output_dim:int, hidden_depth:int, output_mod=None):
    """
    Creates a Multi-layered Perceptron with the specified dimensions and depth.

    Args:
        input_dim (int): The dimension of the input layer.
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): The dimension of the output layer.
        hidden_depth (int): The number of hidden layers in the network.
        output_mods (torch.Module, optional): An optional module to be appended to the output layer, 
            such as an activation function or another layer. Default is None.
    
    Returns:
        nn.Sequential
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
        if output_mod:
            mods.append(output_mod)
        trunk = nn.Sequential(*mods)
        return trunk

def to_np(tensor):
    """
    Converts a PyTorch tensor to a NumPy array.
    """
    if tensor is None:
        return None
    elif tensor.nelement() == 0:
        return np.array([])
    else:
        return tensor.cpu().detach().numpy()