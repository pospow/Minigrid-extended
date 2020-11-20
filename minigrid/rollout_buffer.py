from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch

class RolloutStorage():
    def __init__(self, rollout_size, obs_size):
        self.rollout_size = rollout_size
        self.obs_size = obs_size
        self.reset()

    def insert(self, step, done, action, log_prob, reward, obs):
        self.done[step].copy_(done)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.obs[step].copy_(obs)

    def reset(self):
        self.done = torch.zeros(self.rollout_size, 1)
        self.returns = torch.zeros(self.rollout_size + 1, 1, requires_grad=False)
        self.actions = torch.zeros(self.rollout_size, 1, dtype=torch.int64)
        self.log_probs = torch.zeros(self.rollout_size, 1)
        self.rewards = torch.zeros(self.rollout_size, 1)
        self.obs = torch.zeros(self.rollout_size, self.obs_size)

    def compute_returns(self, gamma):
        self.last_done = (self.done == 1).nonzero().max()
        self.returns[self.last_done + 1] = 0.

        for step in reversed(range(self.last_done + 1)):
            self.returns[step] = self.returns[step + 1] * gamma * (1 - self.done[step]) + self.rewards[step]

    def batch_sampler(self, batch_size, get_old_log_probs=False):
        sampler = BatchSampler(SubsetRandomSampler(range(self.last_done)), batch_size, drop_last=True)

        for indices in sampler:
            if get_old_log_probs:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.log_probs[indices]
            else:
                yield self.actions[indices], self.returns[indices], self.obs[indices]
