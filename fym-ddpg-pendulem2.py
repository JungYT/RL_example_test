import numpy as np
import random
from types import SimpleNamespace as SN
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from fym.core import BaseEnv, BaseSystem


def wrap(angle):
    angle_wrap = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle_wrap


class Pendulum(BaseEnv):
    def __init__(self):
        super().__init__()
        self.th = BaseSystem(np.vstack([0.]))
        self.thdot = BaseSystem(np.vstack([0.]))
        self.g = 10
        self.m = 1.
        self.l = 1.
        self.max_u = 2.
        self.max_speed = 8.

    def set_dot(self, u):
        th, thdot = self.state

        thdot_tmp = -3 * self.g / (2 * self.l) * np.sin(th + np.pi) \
            + 3. / (self.m * self.l ** 2) * u

        self.th.dot = thdot
        self.thdot.dot = np.clip(thdot_tmp, -self.max_speed, self.max_speed)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.05, max_t=20)
        self.pendulum = Pendulum()

    def reset(self):
        super().reset()
        self.pendulum.state = np.vstack((
            np.random.uniform(low=-np.pi, high=np.pi),
            np.random.uniform(low=-1, high=1)
        ))
        th, thdot = self.state
        state = np.hstack([ np.cos(th), np.sin(th), thdot ]).reshape(1, -1)
        return state

    def step(self, u):
        *_, done = self.update(u=u)
        r = self.reward(u)
        th, thdot = self.state
        state = np.hstack([ np.cos(th), np.sin(th), thdot ]).reshape(1, -1)

        return state, r, done

    def set_dot(self, t, u):
        self.pendulum.set_dot(u)

    def reward(self, u):
        th, thdot = self.state
        th = wrap(th)
        r = -th ** 2 - 0.1 * thdot ** 2 - 0.001 * u ** 2
        return r


class ActorNet(nn.Module):
    def __init__(self, cfg):
        super(ActorNet, self).__init__()
        self.lin1 = nn.Linear(cfg.dim_state, cfg.actor_node)
        self.lin2 = nn.Linear(cfg.actor_node, cfg.actor_node)
        self.lin3 = nn.Linear(cfg.actor_node, cfg.dim_action)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(cfg.actor_node)
        self.bn2 = nn.BatchNorm1d(cfg.actor_node)
        self.cfg = cfg

    def forward(self, state):
        x1 = self.relu(self.bn1(self.lin1(state)))
        x2 = self.relu(self.bn2(self.lin2(x1)))
        x3 = self.tanh(self.lin3(x2))
        x_scaled = x3 * self.cfg.action_scaling
        return x_scaled

class CriticNet(nn.Module):
    def __init__(self, cfg):
        super(CriticNet, self).__init__()
        self.lin1 = nn.Linear(cfg.dim_state, cfg.critic_node-cfg.dim_action)
        self.lin2 = nn.Linear(cfg.critic_node, cfg.critic_node)
        self.lin3 = nn.Linear(cfg.critic_node, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(cfg.critic_node-cfg.dim_action)
        self.bn2 = nn.BatchNorm1d(cfg.critic_node)

    def forward(self, state, action):
        x1 = self.relu(self.bn1(self.lin1(state)))
        x_cat = torch.cat((x1, action), dim=1)
        x2 = self.relu(self.bn2(self.lin2(x_cat)))
        x3 = self.lin3(x2)
        return x3

class DDPG:
    def __init__(self, cfg):
        self.memory = deque(maxlen=cfg.memory_size)
        self.behavior_actor = ActorNet(cfg).float()
        self.behavior_critic = CriticNet(cfg).float()
        self.target_actor = ActorNet(cfg).float()
        self.target_critic = CriticNet(cfg).float()
        self.actor_optim = optim.Adam(
            self.behavior_actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optim = optim.Adam(
            self.behavior_critic.parameters(), lr=cfg.critic_lr
        )
        self.mse = nn.MSELoss()
        self.hardupdate(self.target_actor, self.behavior_actor)
        self.hardupdate(self.target_critic, self.behavior_critic)
        self.cfg = cfg

    def get_action(self, state, net='behavior'):
        with torch.no_grad():
            action = self.behavior_actor(torch.FloatTensor(state)) \
                if net == "behavior" \
                else self.target_actor(torch.FloatTensor(state))
        return np.array(np.squeeze(action))

    def memorize(self, item):
        self.memory.append(item)

    def get_sample(self):
        sample = random.sample(self.memory, self.cfg.batch_size)
        state, action, reward, state_next, epi_done = zip(*sample)
        x = torch.FloatTensor(state)
        u = torch.FloatTensor(action)
        r = torch.FloatTensor(reward).view(-1,1)
        xn = torch.FloatTensor(state_next)
        done = torch.FloatTensor(epi_done).view(-1,1)
        return x, u, r, xn, done

    def train(self):
        x, u, r, xn, done = self.get_sample()
        with torch.no_grad():
            action = self.target_actor(xn)
            Qn = self.target_critic(xn, action)
            target = r + (1-done)*self.cfg.discount*Qn
        Q_w_noise_action = self.behavior_critic(x, u)
        self.critic_optim.zero_grad()
        critic_loss = self.mse(Q_w_noise_action, target)
        critic_loss.backward()
        self.critic_optim.step()

        action_wo_noise = self.behavior_actor(x)
        Q = self.behavior_critic(x, action_wo_noise)
        self.actor_optim.zero_grad()
        actor_loss = torch.sum(-Q)
        actor_loss.backward()
        self.actor_optim.step()

        self.softupdate(self.target_actor, self.behavior_actor)
        self.softupdate(self.target_critic, self.behavior_critic)

    def save_params(self, path_save):
        torch.save({
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'behavior_actor': self.behavior_actor.state_dict(),
            'behavior_critic': self.behavior_critic.state_dict()
        }, path_save)

    def set_train_mode(self):
        self.behavior_actor.train()
        self.behavior_critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def set_eval_mode(self):
        self.behavior_actor.eval()
        self.behavior_critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def hardupdate(self, target, behavior):
        target.load_state_dict(behavior.state_dict())
        for target_param, behavior_param in zip(
            target.parameters(),
            behavior.parameters()
        ):
            target_param.data.copy_(behavior_param.data)

    def softupdate(self, target, behavior):
        for target_param, behavior_param in zip(
            target.parameters(),
            behavior.parameters()
        ):
            target_param.data.copy_(
                target_param.data*(1. - self.cfg.softupdate_rate) \
                + behavior_param.data*self.cfg.softupdate_rate
            )

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

def main():
    # agent parameters
    cfg = SN()
    cfg.ddpg = SN()
    cfg.ddpg.dim_state = 3
    cfg.ddpg.dim_action = 1
    cfg.ddpg.action_max = 2.
    cfg.ddpg.action_min = -2.
    cfg.ddpg.action_scaling = 2.
    cfg.ddpg.memory_size = 20000
    cfg.ddpg.actor_lr = 0.0001
    cfg.ddpg.critic_lr = 0.001
    cfg.ddpg.batch_size = 64
    cfg.ddpg.discount = 0.95
    cfg.ddpg.softupdate_rate = 0.001
    cfg.ddpg.terminate_condition = 10
    cfg.ddpg.reward_weight = 20
    cfg.ddpg.reward_max = 210
    # cfg.ddpg.actor_node = 64
    # cfg.ddpg.critic_node = 64
    cfg.ddpg.node_set = [64, 128, 256]
    cfg.ddpg.actor_node = 64
    cfg.ddpg.critic_node = 64

    agent = DDPG(cfg.ddpg)
    n_data = 0

    # noise parameters
    rho = 0.15
    mu = 0
    sigma = 0.2
    dt = 0.1
    noise = OrnsteinUhlenbeckNoise(rho, mu, sigma, dt, 1)

    # environment
    env = Env()

    # display
    test_interval = 10
    n_test = 10
    score = 0
    reward = 0

    for n_epi in range(100):
        x = env.reset()
        noise.reset()
        while True:
            agent.set_eval_mode()
            u = agent.get_action(x) + noise.sample()
            xn, r, done = env.step(u)
            r_train = (r + 8) / 8
            agent.memorize((x.squeeze(), u, r_train, xn.squeeze(), done))
            x = xn
            n_data += 1
            if n_data > 1000:
                agent.set_train_mode()
                agent.train()

            if done:
                break

        if n_epi % test_interval == 0 and n_epi != 0:
            for test in range(n_test):
                x = env.reset()
                while True:
                    agent.set_eval_mode()
                    u = agent.get_action(x)
                    x, r, done = env.step(u)
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







