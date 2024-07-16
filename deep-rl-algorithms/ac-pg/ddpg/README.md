## abstract - high level summary (what, why, outcome)

the paper utilizes a determinstic policy for action selection in a continuous space instead of a stochastic and discretized one, which works well for physics tasks.

## intro - high level overview (focus on the why)

Many tasks operate in continous and high-dimensional actions spaces, where DQN fails miserably. Since DQN uses the argmax of the action-value function, doing so in a continous valued case would require an *interative optimization process* at every step.

Discretizing also does not work well in continous action spaces because of the curse of dimensionality making the algorithm intractable along with the loss of information about the action space structure.

Naive determininistic policy gradients (DPG) is not stable in such domains, for which a replay buffer and a target network is used (from DQN). Instability in DPG is due to the model learning to predict a moving target (bootstrapped target), i.e. the value function, as well as due to the correlation among transitions fed into the model.

## background - mathematical details; pay close attention

$Q^\mu(s_t,a_t)=\E_{r_t,s_{t+1}\sim E}[r(s_t,a_t)+\gamma Q^\mu(s_{t+1},\mu(s_{t+1}))]$

where $\mu(s)$ is the deterministic target policy. The Q-function is denoted with a $\mu$ superscript since $\mu$ is a parameter of the function.

$y_t=r(s_t,a_t)+\gamma Q(s_{t+1},\mu(s_{t+1})\mid\theta^Q)$

## methods - how experiments were performed

Essentially, the critic is a function of the state, and the actor. DPG with neural networks and batch learning for stability is intractable for large networks, which is why minibatches are used.

Replay buffer sampling is uniform, and the samples are removed sequentially when the buffer is full.

"Soft" target updates are used, instead of directly copying the weights of the current network onto the target network, for improved stabilization.

Batch normalization was used as a solution to the variance in features of the state across environments. In low-dimensional case, batch norm was used on all layers prior to the action output.

The exploration problem for deterministic continuous action outputs is solved by adding an additional noise parameter (Ornstein-Uhlenbeck process for exploration effiency in physical control problems) to the actor: $\mu^'(s_t) = \mu(s_t\mid\theta_t^\mu)+\mathcal N$

## results - what are we shooting for (ballpark)

Learning with DDPG on low-dimensional and high-dimensional outputs is equally fast (around 200-300K steps to reach optimal rewards)

## architecture & algos - what are we implementing
Questions to answer - 
- what algorithm?
DDPG (essentially DQN but deterministic actor-critic with OU noise, batch norm, and soft target updates)
- what data structures?
Deque/array for replay buffer, actor and critic networks
- what model architecture?
the networks use relu and batch norm for all hidden layers, the final output layer of the actor is a tanh layer to bound the actions (need to multiply with max action value). the layers except the final layers' weights and biases of both actor and critic networks were uniformly initialized (-1/sqrt(f),1/sqrt(f)) where f is the fan-in (number of inputs). The final layer was uniformly intialized (-3e-3,3e-3).
- what hyperparameters?
```
lr              1e-5 for actor
                1e-4 for critic
gamma           0.99
tau             1e-3
low-dimensional
    fc1         400 neurons
    fc2         300 neurons
    batch_size      64
high-dimensional
    filters     32
    fc1         200
    fc2         200
    batch_size  16
ou_theta        0.15
ou_sigma        0.2
```
- what results to expect?
see above