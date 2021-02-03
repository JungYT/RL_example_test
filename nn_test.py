import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from collections import deque

torch.manual_seed(0)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lin2 = nn.Linear(2,1)
        self.lin3 = nn.Linear(2,1)

    def forward(self, x):
        mean = self.lin2(x)
        std = self.lin3(x)
        return mean, std

net = Net()
a = ([0, -3], [1, -4])
a = torch.FloatTensor(a)
mean, std = net(a)
print(mean, std)

dist = Normal(mean, std)
action = ([0.3], [0.2])
log_prob = dist.log_prob(torch.FloatTensor(action))
print(log_prob)
dist_test = Normal(mean[0], std[0])
print(dist_test.log_prob(torch.FloatTensor(action[0])))
dist_test2 = Normal(mean[1], std[1])

at = dist_test.sample()
at_c = torch.clamp(at, -3, 3)
ar = (at_c.item(), )

at2 = dist_test2.sample()
at_c2 = torch.clamp(at2, -3, 3)
ar2 = (at_c2.item(), )

data = deque()
data.append(ar)
data.append(ar2)

aa = list(data)
print(dist.log_prob(torch.FloatTensor(aa)))
print(dist_test.log_prob(torch.FloatTensor(aa[0])))
print(dist_test2.log_prob(torch.FloatTensor(aa[1])))
print(type(dist))
print(log_prob)
print(log_prob * log_prob)
