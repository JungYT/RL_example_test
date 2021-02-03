import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

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
        self.memory = deque()
        self.gamma = 0.98

    def distribution(self, x):
        prob = self.net(torch.FloatTensor(x))
        distribution = Categorical(prob)

        return distribution

    def action(self, x):
        with torch.no_grad():
            distribution = self.distribution(x)
            action = distribution.sample()
        #log_prob = torch.log(prob[action])

        return action.item()

    def put(self, item):
        self.memory.append(item)

    def get_return(self):
        G = 0

    def replay(self):
        x, u, r = zip(*list(self.memory))
        prob = self.net(torch.FloatTensor(x))
        log_prob = torch.log(prob(x))
        print(log_prob)
        print(x)


    def train(self):
        self.replay()
        '''
        G = 0
        self.optimizer.zero_grad()
        for r, log_prob in self.data[::-1]:
            G = r + self.gamma * G
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()
        self.memory.clear()
        '''


def main():
    env = gym.make('CartPole-v1')
    env.seed(0)
    agent = Agent()
    show_interval = 100
    score = 0

    for n_epi in range(5000):
        x = env.reset()
        while True:
            u = agent.action(x)
            x, r, done, _ = env.step(u)
            agent.put((x, u, r))
            score += r
            if done:
                break

        agent.train()

        if n_epi % show_interval == 0 and n_epi != 0:
            print("# of episode: {},\
                  avg score: {}"
                  .format(n_epi, score / show_interval))
            score = 0

    env.close()


if __name__ == "__main__":
    main()
