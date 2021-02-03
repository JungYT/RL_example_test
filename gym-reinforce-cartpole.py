import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(0)
np.random.seed(0)

class ReinforceNet(nn.Module):
    def __init__(self):
        super(ReinforceNet, self).__init__()

        self.lin1 = nn.Linear(4, 128)
        self.lin2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.softmax(self.lin2(x))

        return x


class Agent():
    def __init__(self):
        self.learning_rate = 0.0002
        self.net = ReinforceNet().float()
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.learning_rate)
        self.data = []
        self.gamma = 0.98

    def distribution(self, x):
        prob = self.net(torch.from_numpy(x).float())
        distribution = Categorical(prob)

        return distribution, prob

    def action(self, x):
        distribution, prob = self.distribution(x)
        action = distribution.sample()
        log_prob = torch.log(prob[action])

        return action, log_prob

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

        return G, loss


def main():
    env = gym.make('CartPole-v1')
    env.seed(0)
    agent = Agent()
    show_interval = 100
    score = 0
    avg_return = 0

    for n_epi in range(5000):
        x = env.reset()
        while True:
            u, log_prob = agent.action(x)
            x, r, done, _ = env.step(u.item())
            agent.put((r, log_prob))
            score += r
            if done:
                break

        G, loss = agent.train()
        avg_return += G

        if n_epi % show_interval == 0 and n_epi != 0:
            print("# of episode: {},\
                  avg score: {},\
                  avg return: {}"
                  .format(n_epi, score / show_interval,
                          avg_return / show_interval))
            score = 0
            avg_return = 0

    env.close()


if __name__ == "__main__":
    main()
