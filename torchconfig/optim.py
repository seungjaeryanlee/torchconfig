import copy

import torch.optim as optim

from torchconfig.filter import filter_args


# Optimizers
def get_optimizer_from_args(params, *args, **kwargs):
    if kwargs["name"] == "SGD":
        return optim.SGD(params, *args, **filter_args(kwargs, optim.SGD))
def get_optimizer_from_dict(params, optimizer_dict):
    return get_optimizer_from_args(params, **optimizer_dict)

# Learning Rate Schedulers
def get_lr_scheduler_from_args(optimizer, *args, **kwargs):
    if kwargs["name"] == "CyclicLR":
        return optim.lr_scheduler.CyclicLR(optimizer, *args, **filter_args(kwargs, optim.lr_scheduler.CyclicLR))
def get_lr_scheduler_from_dict(optimizer, lr_scheduler_dict):
    return get_lr_scheduler_from_args(optimizer, **lr_scheduler_dict)
