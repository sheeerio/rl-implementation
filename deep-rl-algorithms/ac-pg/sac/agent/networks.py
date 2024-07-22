import numpy as np
import torch as T
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as dist

import utils

class TanhTransform(dist.transforms.Transform):
    domain = dist.constraints.real
    codomain = dist.constraints.interval(-1., 1.)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)
    
    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())
    
    def __eq__(self, other):
        return isinstance(other, TanhTransform)
    
    def _call(self, x):
        return x.tanh()
    
    def _inverse(self, y):
        return self.atanh(y)
    
    def log_abs_det_jacobian(self, x):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(dist.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = dist.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dict, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class DiagGaussianActor(nn.Module):
    """
    Diagonal Gaussian policy.
    """
    def __init__(self, obs_dim, hidden_dim, action_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
        
        self.apply(utils.weight_init)
    
    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = T.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * \
            (log_std + 1)
        
        std = log_std.exp()
        dist = SquashedNormal(mu, std)

        return dist

class DoubleQCritic(nn.Module):
    """Critic network, employs double Q-learning"""
    def __init__(self, obs_dim, hidden_dim, action_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = T.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2