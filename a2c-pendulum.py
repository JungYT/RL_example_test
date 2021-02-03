import numpy as np
from fym.core import BaseEnv, BaseSystem
import fym.logging as logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class Pendulum(BaseEnv):
    def __init__(self, th_init):
        super().__init__()
        self.th = BaseSystem(th_init)
        self.thdot = BaseSystem(0)
        self.g = 10
        self.m = 1.
        self.l = 1.
        self.max_u = 2.
        self.max_speed = 8

    def set_dot(self, u):
        th, thdot = self.state

        th = (th + np.pi) % (2 * np.pi) - np.pi

        self.th.dot = thdot
        thdot_temp = -3 * self.g / (2 * self.l) * np.sin(th + np.pi) \
            + 3. / (self.m * self.l ** 2) * u
        self.thdot.dot = np.clip(thdot_temp, -self.max_speed, self.max_speed)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.data = []
        self.gamma = 0.99

        self.lin1 = nn.Linear(2, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin_mean = nn.Linear(64, 1)
        self.lin_sig = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.0005)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        mean = self.tanh(self.lin_mean(x))
        sig = self.softplus(self.lin_sig(x))

        return mean, sig

    def put_data(self, item):
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


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.05, max_t=20)
        self.pendulum = Pendulum(np.pi)

    def reset(self):
        super().reset()
        return self.state

    def step(self, u):
        *_, done = self.update(u=u)
        r = self.reward(u)

        return self.state, r, done

    def step_test(self, u):
        *_, done = self.update(u=u)
        r = self.reward(u)

        info = {
            "input": u,
            "reward": r,
        }
        self.logger.record(**info)

        return self.state, done

    def set_dot(self, t, u):
        self.pendulum.set_dot(u)

    def reward(self, u):
        th, thdot = self.state

        r = -th ** 2 - 0.1 * thdot **2 - 0.001 * u ** 2
        return r



def main():
    env = Env()
    policy = Actor()
    show_result = 20

    for n_epi in range(5000):
        obs = env.reset()
        #env.render()
        while True:
            mean, sig = policy(torch.from_numpy(obs).float())
            policy_prob = Normal(mean, sig)
            u = policy_prob.sample()
            obs, r, done = env.step(u)
            log_policy = policy_prob.log_prob(u)
            policy.put_data((r, log_policy))
            if done:
                break

        G, loss = policy.train()
        if n_epi % show_result == 0 and n_epi != 0:
            print("# of episode: {}, return: {}, loss: {}".format(n_epi,
                                                                  G.item(),
                                                                  loss.item()))
    env.logger = logging.Logger()
    obs = env.reset()
    while True:
        mean, sig = policy(torch.from_numpy(obs).float())
        obs, done = env.step_test(mean.item())
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
