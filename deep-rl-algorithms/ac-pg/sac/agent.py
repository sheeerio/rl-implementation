import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, ValueNetwork, QNetwork
from utils import update
from buffer import ReplayBuffer

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        
        self.gamma = args.gamma
        self.tau = args.tau
        self.learn_cntr = 0
        self.alpha = args.alpha
        self.batch_size = args.batch_size
        self.scale = args.reward_scale
        self.memory = ReplayBuffer(args.max_size, num_inputs, action_space.shape[0])

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = T.device("mps")
        self.critic1 = QNetwork(num_inputs, action_space.shape[0], lr=args.lr, name='critic1', hidden_dims=args.hidden_size)
        self.critic2 = QNetwork(num_inputs, action_space.shape[0], lr=args.lr, name='critic2', hidden_dims=args.hidden_size)
        self.value = ValueNetwork(num_inputs, name='value', lr=args.lr)
        self.value_ = ValueNetwork(num_inputs, name='target_value', lr=args.lr)
        update(self.value_, self.value)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -T.prod(T.tensor(action_space.shape)).to(self.device).item()
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = T.optim.Adam([self.log_alpha], lr=args.lr)
        
        self.policy = ActorNetwork(num_inputs, action_space.shape[0], name='actor', lr=args.lr, action_space=action_space)
    
    def choose_action(self, obs, evaluate=False):
        obs = T.tensor(obs, dtype=T.float).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample_normal(obs, reparameterize=False)
        else:
            _, _, action = self.policy.sample_normal(obs, reparameterize=False)
        
        return action.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, done, state_)
    
    def save_models(self):
        print('... saving models ...')
        self.policy.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.value.save_checkpoint()
        self.value_.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.policy.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.value.load_checkpoint()
        self.value_.load_checkpoint()
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, states_, rewards, dones, actions = self.memory.sample_transition(self.batch_size)
        reward = T.tensor(rewards, dtype=T.float).to(self.critic1.device)
        done = T.tensor(dones, dtype=T.bool).to(self.critic1.device)
        state_ = T.tensor(states_, dtype=T.float).to(self.critic1.device)
        state = T.tensor(states, dtype=T.float).to(self.critic1.device)
        action = T.tensor(actions, dtype=T.float).to(self.critic1.device)

        value = self.value(state).view(-1)
        value_ = self.value_(state_).view(-1)
        value_[done] = 0.0
       
        actions, log_probs, _ = self.policy.sample_normal(state, reparameterize=False)
        #actions, log_probs, _ = self.policy.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs, _ = self.policy.sample_normal(state, reparameterize=True)
        #actions, log_probs = self.policy.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward(state, actions)
        q2_new_policy = self.critic2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.policy.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.policy.optimizer.step()

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic1.forward(state, action).view(-1)
        q2_old_policy = self.critic2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        update(self.value_, self.value)