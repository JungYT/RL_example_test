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

        self.lin1 = nn.Linear(1,1, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        return x

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        self.lin1 = nn.Linear(1,1, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        return x

class Agent():
    def __init__(self):
        self.actor = ActorNet().float()
        actor_param = list(self.actor.parameters())
        print('actor parameter: ', get_tensor_info(actor_param[0]))
        self.critic = CriticNet().float()
        critic_param = list(self.critic.parameters())
        print('critic parameter: ', get_tensor_info(critic_param[0]))
        self.data = []

    def action(self, x):
        with torch.no_grad():
            a = self.actor(x)
        return a

    def put(self, item):
        self.data.append(item)

    def get_target(self, x, a, r, xn):
        with torch.no_grad():
            Vn = self.critic(xn)
            print('Vn: ', get_tensor_info(Vn))
            target = r + Vn
            Vp = self.critic(x)
            advantage = target - Vp

        return target, advantage

    def replay(self):
        x, u, r, xn = zip(*self.data)
        x = torch.FloatTensor(x)
        u = torch.FloatTensor(u)
        r = torch.FloatTensor(r)
        xn = torch.FloatTensor(xn)

        xr = self.actor(x)

        log_prob = 2 * xr * u
        Vp = self.critic(x)
        #print('Vp: ', get_tensor_info(Vp))
        target, advantage = self.get_target(x, u, r, xn)

        return log_prob, Vp, target, advantage

    def train(self):
        log_prob, Vp, target, advantage = self.replay()
        critic_loss = (Vp - target) ** 2
        actor_loss = log_prob * advantage
        critic_loss.backward()
        actor_loss.backward()
        actor_param = list(self.actor.parameters())
        print('actor parameter after backward: ', get_tensor_info(actor_param[0]))
        print('target:', get_tensor_info(target))
        print('Vp:', get_tensor_info(Vp))
        critic_param = list(self.critic.parameters())
        print('critic parameter after backward: ', get_tensor_info(critic_param[0]))


def system(x, a):
    xn = 2 * x + a
    r = 1
    return xn, r


def get_tensor_info(tensor):
    info = []
    for name in ['requires_grad', 'is_leaf', 'grad']:
        info.append(f'{name}({getattr(tensor, name)})')
    info.append(f'tensor({str(tensor)})')
    return ' '.join(info)

agent = Agent()
x = 3
a = agent.action(torch.FloatTensor([x]))
print('a: ', get_tensor_info(a))
xn, r = system(x, a)
agent.put((x, a, r, xn))
agent.train()


