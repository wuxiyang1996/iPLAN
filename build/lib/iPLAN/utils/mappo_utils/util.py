import copy
import numpy as np
import math
import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape