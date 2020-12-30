import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from minigrid.utils import count_model_params


class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.num_actions = num_actions
        self.fc = nn.Sequential(nn.Linear(num_inputs, hidden_dim), nn.Tanh(),
                                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                nn.Linear(hidden_dim, num_actions))

    def forward(self, state):
        x = self.fc(state)
        return x


class Policy():
    def __init__(self, num_inputs, num_actions, hidden_dim, learning_rate, batch_size,
                 policy_epochs, entropy_coef=0.001):
        self.actor = ActorNetwork(num_inputs, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef
        self.name = 'PG'

    def act(self, state):
        logits = self.actor.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, state, action):
        logits = self.actor.forward(state)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(action.squeeze(-1)).view(-1, 1)
        entropy = dist.entropy().view(-1, 1)
        return log_prob, entropy

    def update(self, rollouts):
        for epoch in range(self.policy_epochs):
            data = rollouts.batch_sampler(self.batch_size)

            for sample in data:
                actions_batch, returns_batch, obs_batch = sample

                log_probs_batch, entropy_batch = self.evaluate_actions(obs_batch, actions_batch)

                policy_loss = -torch.mean(log_probs_batch * returns_batch)
                entropy_loss = -torch.mean(entropy_batch)

                loss = policy_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()

    @property
    def num_params(self):
        return count_model_params(self.actor)