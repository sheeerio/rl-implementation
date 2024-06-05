# BATCH_SIZE = 128
# GAMMA = 0.99
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 1000
# TAU = 0.005
# LR = 1e-4

# n_actions = env.action_space.n
# state, info = env.reset()
# n_obs = len(state)


# class DQN(nn.Module):
#     def __init__(self, n_obs, n_actions):
#         super(DQN, self).__init__()
#         self.l1 = nn.Linear(n_obs, 128)
#         self.l2 = nn.Linear(128, 128)
#         self.l3 = nn.Linear(128, n_actions)
    
#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         return self.l3(x)
    
# policy_net = DQN(n_obs, n_actions)
# target_net = DQN(n_obs, n_actions)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# memory = ReplayMemory(10000)
# steps_done = 0

# def select_action(state):
#     global steps_done
#     sample = random.random()
#     epsilon = EPS_END + (EPS_START - EPS_END) * \
#         math.exp(-1 * steps_done / EPS_DECAY)
#     steps_done +=1
#     if sample > epsilon:
#         # max action
#         with torch.no_grad():
#             return policy_net(state).max(1).indices.view(1,1)
#     else:
#         # random action
#         return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

# episodes_durations = []

# def optimize_model():
#     if len(memory) < BATCH_SIZE:
#         return

# loss = r + gamma * q_target - q_val
# class DQN(nn.Module):
#     def __init__(self, state_dim:int, action_dim:int, qnet:nn.Module, qnet_target:nn.Module,lr:float, gamma:float, epsilon:float):
#         super(DQN, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.qnet = qnet
#         self.qnet_target = qnet_target
#         self.lr = lr
#         self.gamma = gamma
#         self.opt = optim.Adam(params=self.qnet.parameters(), lr=lr)
#         self.register_buffer('epsilon', torch.ones(1)*epsilon)
#         self.criteria = nn.SmoothL1Loss()
    
#     def get_action(self, state):
#         qs = self.qnet(state)
#         prob = np.random.uniform(0.0,1.0,1)
#         if torch.from_numpy(prob).float() <= self.epsilon: # random
#             action = np.random.choice(range(self.action_dim))
#         else: # greedy
#             action = qs.argmax(dim=-1)
#         return int(action)
    
#     def update(self, s, a, r, s_, d):
#         with torch.no_grad():
#             q_max, _ = self.qnet_target(s_).max(dim=-1, keepdims=True)
#             q_target = r + self.gamma * q_max * (1 - d)
#             q_target = q_target.detach()
        
#         q_pred = self.qnet(s)[0,a]
#         loss = self.criteria(q_pred, q_target)
#         self.opt.zero_grad()
#         loss.backward()
#         self.opt.step()

# # HYPER PARAMS
# LR = 1e-4 # NEED TO CHANGE
# BATCH_SIZE = 256
# GAMMA = 1.0
# MEMORY_SIZE = 50000
# TOTAL_EPS = 3000
# EPS_MAX = 0.08
# EPS_MIN = 0.01
# SAMPLING_UNTIL = 2000
# TARGET_UPDATE_INTERVAL = 10

# qnet = QNet(4, 2)
# qnet_target = QNet(4, 2)

# qnet_target.load_state_dict(qnet.state_dict())
# agent = DQN(4, 2, qnet, qnet_target, lr=LR, gamma=GAMMA, epsilon=1.0)

# memory = ReplayMemory(MEMORY_SIZE)

# for e in range(TOTAL_EPS):
#     eps = max(EPS_MIN, EPS_MAX - EPS_MIN * (e / 200))
#     agent.epsilon = torch.tensor(eps)
#     s = env.reset()
#     cum_rew = 0

#     while True:
#         s = torch.tensor(s, dtype=torch.float32)
#         a = agent.get_action(s)
#         s_,r,d,info = env.step(a)
#         experience = (s, 
#                       torch.tensor(a,dtype=torch.float32),
#                       torch.tensor(r / 100.0, dtype=torch.float32),
#                       torch.tensor(s_, dtype=torch.float32),
#                       torch.tensor(d, dtype=torch.float32)
#                       )
#         memory.push(experience)
#         s = s_
#         cum_rew += r
#         if d:
#             break

#     if len(memory) >= SAMPLING_UNTIL:
#         batch = np.random.choice(len(ReplayMemory), BATCH_SIZE, replace=False)

#         state_batch = self.
#         agent.update(state_batch,action_batch,reward_batch,next_state_batch,done_batch)

#     if e % TARGET_UPDATE_INTERVAL == 0:
#         qnet_target.load_state_dict(qnet.state_dict())
    
#     if e % 100 == 0:
#         print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | EPsilon : {:.3f}".format(e,cum_rew,eps))
# print('Done')
# env.close()

