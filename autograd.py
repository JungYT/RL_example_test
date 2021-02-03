import numpy as np
import torch

def get_tensor_info(tensor):
    info = []
    for name in ['requires_grad', 'is_leaf', 'grad']:
        info.append(f'{name}({getattr(tensor, name)})')
    info.append(f'tensor({str(tensor)})')
    return ' '.join(info)


x = torch.tensor(3.0, requires_grad=True)
xn = torch.tensor(5.0, requires_grad=True)
w1 = torch.tensor(7.0, requires_grad=True)
w2 = torch.tensor(4.0, requires_grad=True)

def action(state):
    return w1 * state

def value(state):
    return w2 * state

a = action(x)
v = value(x)
with torch.no_grad():
    t = value(xn)
    A = value(xn) - value(x)

action_loss = a * A
critic_loss = (t -  v) ** 2 / 2

action_loss.backward()
print('x after backward of action loss:', get_tensor_info(x))
print('w1 after backward of action loss:', get_tensor_info(w1))
print('t:', get_tensor_info(t))
print('A:', get_tensor_info(A))

a = action(xn)
with torch.no_grad():
    t = value(xn)
    A = value(xn) - value(x)
action_loss = a * A
action_loss.backward()
print('x after 2backward of action loss:', get_tensor_info(x))
print('w1 after 2backward of action loss:', get_tensor_info(w1))

x.grad.zero_()
critic_loss.backward()
print('x after backward of action loss:', get_tensor_info(x))
print('w2 after backward of critic loss:', get_tensor_info(w2))

