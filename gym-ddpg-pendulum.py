import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

torch.manual_seed(0)
np.random.seed(0)

class ActorNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNet, self).__init__()

        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_bound = 2.0

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.tanh(self.lin4(x)) * self.action_bound

        return x


class CriticNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNet, self).__init__()

        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64 + action_size, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.xa_debug = []
        self.x_debug = []
        self.a_debug = []

    def forward(self, xa):
        x, a = xa
        self.xa_debug = xa
        self.x_debug = x
        self.a_debug = a
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(torch.cat([x, a], 1)))
        x = self.relu(self.lin3(x))
        x = self.lin4(x)

        return x


class OrnsteinUhlenbeckNoise():
    def __init__(self, rho, mu, sigma, dt, size, x0=None):
        self.rho = rho
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset()

    def reset(self):
        self.x = self.x0 if self.x0 is not None else np.zeros(self.size)

    def sample(self):
        x = self.x + self.rho * (self.mu - self.x) * self.dt \
            + np.sqrt(self.dt) * self.sigma * np.random.normal(size=self.size)
        self.x = x

        return x


class DDPG():
    def __init__(self, state_size, action_size):
        # learning hyper parameters
        self.gamma = 0.95
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.001

        # memory parameters
        self.buffer_size = 20000
        self.batch_size = 64

        # setting
        self.memory = deque(maxlen=self.buffer_size)
        self.behavior_actor = ActorNet(state_size, action_size).float()
        self.behavior_critic = CriticNet(state_size, action_size).float()
        self.target_actor = ActorNet(state_size, action_size).float()
        self.target_critic = CriticNet(state_size, action_size).float()
        self.actor_optim = optim.Adam(self.behavior_actor.parameters(),
                                       lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.behavior_critic.parameters(),
                                       lr=self.critic_lr)
        self.mse = nn.MSELoss()
        hard_update(self.target_actor, self.behavior_actor)
        hard_update(self.target_critic, self.behavior_critic)

    def action_train(self, x):
        with torch.no_grad():
            u = self.behavior_actor(torch.FloatTensor(x))
            u_clamped = torch.clamp(u, -self.behavior_actor.action_bound,
                                    self.behavior_actor.action_bound)

        return (u_clamped.item(), )

    def action_test(self, x):
        u = self.target_actor(torch.FloatTensor(x))
        u_clamped = torch.clamp(u, -self.target_actor.action_bound,
                                self.target_actor.action_bound)

        return (u_clamped.item(), )

    def memorize(self, item):
        self.memory.append(item)

    def sample(self):
        sample = random.sample(self.memory, self.batch_size)
        x, u, r, xn, done = zip(*sample)
        x = torch.FloatTensor(x)
        u = torch.FloatTensor(u)
        r = torch.FloatTensor(r).view(-1, 1)
        xn = torch.FloatTensor(xn)
        done = torch.FloatTensor(done).view(-1, 1)

        return x, u, r, xn, done

    def train(self):
        x, u, r, xn, done = self.sample()

        with torch.no_grad():
            action = self.target_actor(xn)
            Qn = self.target_critic([xn, action])
            target = r + (1 - done) * self.gamma * Qn

        Q_noise = self.behavior_critic([x, u])

        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_noise, target)
        critic_loss.backward()
        self.critic_optim.step()

        Q = self.behavior_critic([x, self.behavior_actor(x)])

        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        soft_update(self.target_actor, self.behavior_actor, self.tau)
        soft_update(self.target_critic, self.behavior_critic, self.tau)


def soft_update(target, behavior, tau):
    for t_param, b_param in zip(target.parameters(), behavior.parameters()):
        t_param.data.copy_(t_param.data * (1.0 - tau) + b_param.data * tau)

def hard_update(target, behavior):
    for t_param, b_param in zip(target.parameters(), behavior.parameters()):
        t_param.data.copy_(b_param.data)

def main():
    # agent parameters
    state_size = 3
    action_size = 1
    agent = DDPG(state_size, action_size)
    n_data = 0

    # noise parameters
    rho = 0.15
    mu = 0
    sigma = 0.2
    dt = 0.1
    noise = OrnsteinUhlenbeckNoise(rho, mu, sigma, dt, action_size)

    # environment
    env = gym.make('Pendulum-v0')
    env.seed(0)

    # display
    test_interval = 10
    n_test = 10
    score = 0
    reward = 0

    for n_epi in range(1000):
        x = env.reset()
        noise.reset()
        while True:
            u = agent.action_train(x) + noise.sample()
            xn, r, done, _ = env.step(u)
            r_train = (r + 8) / 8
            agent.memorize((x, u, r_train, xn, done))
            n_data += 1
            x = xn
            if n_data > 1000:
                agent.train()

            if done:
                break

        if n_epi % test_interval == 0 and n_epi != 0:
            for test in range(n_test):
                x = env.reset()
                while True:
                    u = agent.action_test(x)
                    x, r, done, _ = env.step(u)
                    r_train = (r + 8) / 8
                    score += r_train
                    reward += r
                    if done:
                        break

            print("# of episode: {},\
                    avg score: {},\
                  avg return: {}".format(n_epi, score / n_test,
                                         reward / n_test))
            score = 0
            reward = 0

    env.close()


if __name__ == "__main__":
    main()
