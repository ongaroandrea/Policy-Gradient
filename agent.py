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

        self.sigmasquared = 5

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
        normal_dist = ...
        
        return normal_dist


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        self.states, self.action_probs, self.rewards = [], [], []

        discounted_rewards = discount_rewards(rewards, self.gamma)

        # T1 b) TODO
        discounted_rewards = ...

        # T1 c) TODO
        discounted_rewards = ...

        # Compute the optimization term, loss function to optimize (T1) TODO
        estimated_loss_batch =...

        # estimated_loss = torch.sum(estimated_loss_batch)
        # Using the mean is more stable as it results in more consistent and "predictable" update sizes (different episodes have different number of timesteps)
        estimated_loss = torch.mean(estimated_loss_batch)

        # Compute the gradients of loss w.r.t. network parameters (T1) TODO
        ...

        # Update network parameters using self.optimizer and zero gradients (T1) TODO
        ...

        return        

    def get_action(self, observation, episode_number=0, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        normal_dist = self.policy.forward(x, episode_number=episode_number)

        # Return mean if evaluation, else sample from the distribution returned by the policy (T1) TODO
        if evaluation:
            return normal_dist.mean, None
        else:
            # Sample from the distribution (T1) TODO
            action = ...
            # Calculate the log probability of the action (T1) TODO
            action_log_prob = ...

        return action, action_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)

        # They are log_action probabilities actually
        self.action_probs.append(action_prob)

        self.rewards.append(torch.Tensor([reward]))

