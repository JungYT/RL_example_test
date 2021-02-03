import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ReinforceNet(nn.Module):
    def __init__(self):
        super(ReinforceNet, self).__init__()

        self.lin1 = nn.Linear(3, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin_mean = nn.Linear(16, 1)
        self.lin_sig = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.action_bound = 2.0

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        mean = self.action_bound * self.tanh(self.lin_mean(x))
        sig = self.softplus(self.lin_sig(x))

        return mean, sig


class Agent():
    def __init__(self):
        self.gamma = 0.95
        self.learning_rate = 0.0001
        self.net = ReinforceNet().float()
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate)
        self.data = []
        self.sig_bound = [1e-3, 1.0]

    def distribution(self, x):
        mean, sig = self.net(torch.from_numpy(x).float())
        sig_clamped = torch.clamp(sig, self.sig_bound[0], self.sig_bound[1])
        distribution = Normal(mean, sig_clamped)

        return distribution

    def action(self, x):
        distribution = self.distribution(x)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        action_clamped = torch.clamp(action,
                                  -self.net.action_bound,
                                    self.net.action_bound)

        return (action_clamped.item(),), log_prob

    def put(self, item):
        self.data.append(item)

    def train(self):
        G = 0
        self.optimizer.zero_grad()
        for r, log_prob in self.data[::-1]:
            G = r + self.gamma * G
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    env = gym.make('Pendulum-v0')
    agent = Agent()
    show_interval = 10
    score = 0

    for n_epi in range(2000):
        x = env.reset()
        while True:
            u, log_prob = agent.action(x)
            x, r, done, _ = env.step(u)
            r_train = (r + 8) / 8
            agent.put((r_train, log_prob))
            score += r_train
            if done:
                break

        agent.train()

        if n_epi % show_interval == 0 and n_epi != 0:
            print("# of episode: {},\
                  avg score: {}"
                  .format(n_epi, score / show_interval))
            score = 0
            avg_return = 0

    env.close()

if __name__ == "__main__":
    main()
