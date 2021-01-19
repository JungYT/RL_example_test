import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        self.lin1 = nn.Linear(3, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.mean = nn.Linear(16, 1)
        self.std = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.action_bound = 2.0

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        mean = self.tanh(self.mean(x)) * self.action_bound
        std = self.softplus(self.std(x))

        return mean, std


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        self.lin1 = nn.Linear(3, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.identity = nn.Identity()
        self.mse = nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.lin4(x)

        return x


class Agent():
    def __init__(self):
        self.gamma = 0.95
        self.batch_size = 32
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.std_bound = [1e-2, 1.0]

        self.actor_net = ActorNet().float()
        self.critic_net = CriticNet().float()
        self.actor_optim = optim.Adam(self.actor_net.parameters(),
                                      lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),
                                       lr=self.critic_lr)
        self.data = []

    def distribution(self, x):
        mean, std = self.actor_net(torch.from_numpy(x).float())
        std_clamped = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        distribution = Normal(mean, std_clamped)

        return distribution

    def action(self, x):
        distribution = self.distribution(x)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        action_clamped = torch.clamp(action, -self.actor_net.action_bound,
                                     self.actor_net.action_bound)

        return (action_clamped.item(),), log_prob

    def put(self, item):
        self.data.append(item)

    def calculate_loss(self):
        actor_loss = 0
        critic_loss = 0
        critic_target_set = []
        critic_output_set = []
        for x, log_prob, r, xn, done in self.data:
            Vp = self.critic_net(torch.from_numpy(x).float())
            Vn = self.critic_net(torch.from_numpy(xn).float())
            if done:
                target = r
                advantage = target - Vp
            else:
                target = r + self.gamma * Vn
                advantage = target - Vp
            actor_loss += -log_prob * advantage
            critic_loss += advantage ** 2
        critic_loss = critic_loss / self.batch_size

        return actor_loss, critic_loss

    def train(self):
        actor_loss, critic_loss = self.calculate_loss()

        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.data = []


    """
    def train(self):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        for x, log_prob, r, xn, done in self.data:
            Vp = self.critic_net(torch.from_numpy(x).float())
            Vn = self.critic_net(torch.from_numpy(xn).float())
            if done:
                target = r
                advantage = target - Vp
            else:
                target = r + self.gamma * Vn
                advantage = target - Vp
            actor_loss = -log_prob * advantage
            #critic_loss = self.critic_net.mse(Vp, torch.tensor(target).float())
            critic_loss = advantage ** 2 / (self.batch_size)
            critic_loss.backward(retain_graph=True)
            actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        self.data = []
    """


def main():
    env = gym.make('Pendulum-v0')
    agent = Agent()
    show_interval = 10
    score = 0
    avg_return = 0

    for n_epi in range(1000):
        x = env.reset()
        while True:
            u, log_prob = agent.action(x)
            xn, r, done, _ = env.step(u)
            r_train = (r + 8) / 8
            agent.put((x, log_prob, r_train, xn, done))
            x = xn
            score += r_train
            if len(agent.data) == agent.batch_size:
                agent.train()

            if done:
                break

        if n_epi % show_interval == 0 and n_epi != 0:
            print("# of episode: {},\
                    avg score: {}".format(n_epi, score / show_interval))
            score = 0

    env.close()


if __name__ == "__main__":
    main()
