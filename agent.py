import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards
import sys
import math

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):        
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)

        self.sigmasquared = 5 # Initial value of sigma squared. This is can be changed to a learnable parameter if needed

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, episode_number=0):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        # --- Instantiate and return a normal distribution with mean as network output
        # T1 TODO
        # Why normal distribution? Because we want to sample actions from a continuous action space
        # We use the square root of the variance because the normal distribution takes the standard deviation as input
        return Normal(action_mean, torch.sqrt(torch.tensor(self.sigmasquared))) 


class Agent(object):
    def __init__(self, policy, algorithm = 'basic'):
        """
        Initialize REINFORCE agent with different variants
        Args:
            policy: Policy network
            algorithm: 'basic', 'constant_baseline', or 'normalized'
            baseline_value: Constant baseline value (used only if algorithm='constant_baseline')
        """
                
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.baseline_value = 20
        self.algorithm = algorithm

    def basic_reinforce(self, action_probs, discounted_rewards):
        """
        (a) Basic REINFORCE without baseline
        """
        return -action_probs * discounted_rewards

    def constant_baseline_reinforce(self, action_probs, discounted_rewards):
        """
        (b) REINFORCE with constant baseline
        """
        return -action_probs * (discounted_rewards - self.baseline_value)

    def normalized_reinforce(self, action_probs, discounted_rewards):
        """
        (c) REINFORCE with normalized rewards
        """
        # Normalize rewards to zero mean and unit variance
        normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8) # + 1e-8 to avoid division by zero
        return -action_probs * normalized_rewards
    
    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        self.states, self.action_probs, self.rewards = [], [], []

        discounted_rewards = discount_rewards(rewards, self.gamma)

        # Compute the discounted rewards (T1) TODO
        if self.algorithm == 'basic':
            estimated_loss_batch = self.basic_reinforce(action_probs, discounted_rewards)
        elif self.algorithm == 'constant_baseline':
            # T1 b) TODO
            estimated_loss_batch = self.constant_baseline_reinforce(action_probs, discounted_rewards)
        elif self.algorithm == 'normalized':
            # T1 c) TODO
            estimated_loss_batch = self.normalized_reinforce(action_probs, discounted_rewards)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")


        # estimated_loss = torch.sum(estimated_loss_batch)
        # Using the mean is more stable as it results in more consistent and "predictable" update sizes (different episodes have different number of timesteps)
        estimated_loss = torch.mean(estimated_loss_batch)

        # Compute the gradients of loss w.r.t. network parameters (T1) TODO
        estimated_loss.backward()

        # Update network parameters using self.optimizer and zero gradients (T1) TODO
        self.optimizer.step()
        # Zero the gradients
        self.optimizer.zero_grad()

        return        

    def get_action(self, observation, episode_number=0, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        normal_dist = self.policy.forward(x, episode_number=episode_number)

        # Return mean if evaluation, else sample from the distribution returned by the policy (T1) TODO
        if evaluation:
            return normal_dist.mean, None
        else:
            # Sample from the distribution (T1) TODO
            action = normal_dist.sample()
 
            # Calculate the log probability of the action (T1) TODO
            action_log_prob = normal_dist.log_prob(action)

        return action, action_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)

        # They are log_action probabilities actually
        self.action_probs.append(action_prob)

        self.rewards.append(torch.Tensor([reward]))

