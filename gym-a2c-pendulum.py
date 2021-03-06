import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

torch.manual_seed(0)
np.random.seed(0)

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
        self.memory = deque(maxlen=self.batch_size)

    def distribution(self, x):
        mean, std = self.actor_net(x)
        std_clamped = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        distribution = Normal(mean, std_clamped)

        return distribution

    def action(self, x):
        with torch.no_grad():
            distribution = self.distribution(torch.FloatTensor(x))
            action = distribution.sample()
            action_clamped = torch.clamp(action, -self.actor_net.action_bound,
                                     self.actor_net.action_bound)

        return (action_clamped.item(),)

    def get_target_advantage(self, x, r, xn, done):
        with torch.no_grad():
            Vp = self.critic_net(x)
            Vn = self.critic_net(xn)
            target = r + (1 - done) * self.gamma * Vn
            advantage = target - Vp

        return target, advantage

    def put(self, item):
        self.memory.append(item)

    def replay(self):
        x, u, r, xn, done = zip(*list(self.memory))
        x = torch.FloatTensor(x)
        u = torch.FloatTensor(u)
        r = torch.FloatTensor(r).view(-1,1)
        xn = torch.FloatTensor(xn)
        done = torch.FloatTensor(done).view(-1,1)

        distribution = self.distribution(x)
        log_prob = distribution.log_prob(u)
        Vp = self.critic_net(x)

        target, advantage = self.get_target_advantage(x, r, xn, done)

        return log_prob, Vp, target, advantage

    def train(self):
        log_prob, Vp, target, advantage = self.replay()

        self.critic_optim.zero_grad()
        critic_loss = self.critic_net.mse(Vp, target)
            #critic_loss = torch.norm(Vp - target) / self.batch_size

        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-log_prob * advantage)
        actor_loss.backward()
        self.actor_optim.step()

    """
    def train(self):
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        for x, log_prob, r, xn, done in self.data:
            Vp = self.critic_net(torch.from_numpy(x).float())
            Vn = self.critic_net(torch.from_numpy(xn).float())
            if done:
                target = r
            else:
                target = r + self.gamma * Vn
            advantage = target - Vp
            critic_loss = self.critic_net.mse(Vp, torch.tensor(target).float())
            #critic_loss = torch.square(advantage)
            critic_loss.backward(retain_graph=True)
            actor_loss = -log_prob * advantage
            actor_loss.backward()
        self.critic_optim.step()
        self.actor_optim.step()
    """

def get_tensor_info(tensor):
    info = []
    for name in ['requires_grad', 'is_leaf', 'grad']:
        info.append(f'{name}({getattr(tensor, name)})')
    info.append(f'tensor({str(tensor)})')
    return ' '.join(info)

def main():
    env = gym.make('Pendulum-v0')
    env.seed(0)
    agent = Agent()
    show_interval = 50
    score = 0
    reward = 0

    for n_epi in range(10000):
        x = env.reset()
        n_data = 0
        while True:
            u = agent.action(x)
            xn, r, done, _ = env.step(u)
            r_train = (r + 8) / 8
            agent.put((x, u, r_train, xn, done))
            n_data += 1
            x = xn
            score += r_train
            reward += r
            if n_data == agent.batch_size:
                agent.train()
                n_data = 0

            if done:
                break

        if n_epi % show_interval == 0 and n_epi != 0:
            print("# of episode: {},\
                    avg score: {},\
                  avg return: {}".format(n_epi, score / show_interval,
                                         reward / show_interval))
            score = 0
            reward = 0

    env.close()


if __name__ == "__main__":
    main()
