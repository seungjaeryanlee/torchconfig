import torch.optim as optim

from torchconfig.filter import get_subdict, filter_args


# Optimizers
NAME_TO_OPTIMIZER = {
    "ASGD": optim.ASGD,
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "Adamax": optim.Adamax,
    "LBFGS": optim.LBFGS,
    "RMSprop": optim.RMSprop,
    "Rprop": optim.Rprop,
    "SGD": optim.SGD,
    "SparseAdam": optim.SparseAdam,
}


def get_optimizer_from_args(params, ignore_cases=False, *args, **kwargs):
    if ignore_cases:
        name = list(get_subdict(kwargs, ["name"], ignore_cases).values())[0]
        optimizer_func = NAME_TO_OPTIMIZER[name]
    else:
        optimizer_func = NAME_TO_OPTIMIZER[kwargs["name"]]
    return optimizer_func(params, *args, **filter_args(kwargs, optimizer_func, ignore_cases))


def get_optimizer_from_dict(params, optimizer_dict, ignore_cases=False):
    return get_optimizer_from_args(params, **optimizer_dict, ignore_cases=ignore_cases)


# Learning Rate Schedulers
NAME_TO_LR_SCHEDULER = {
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "CyclicLR": optim.lr_scheduler.CyclicLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR,
    "MultiStepLR": optim.lr_scheduler.MultiStepLR,
    "MultiplicativeLR": optim.lr_scheduler.MultiplicativeLR,
    "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "StepLR": optim.lr_scheduler.StepLR,
}


def get_lr_scheduler_from_args(optimizer, ignore_cases=False, *args, **kwargs):
    if ignore_cases:
        name = list(get_subdict(kwargs, ["name"], ignore_cases).values())[0]
        lr_scheduler_func = NAME_TO_LR_SCHEDULER[name]
    else:
        lr_scheduler_func = NAME_TO_LR_SCHEDULER[kwargs["name"]]
    return lr_scheduler_func(optimizer, *args, **filter_args(kwargs, lr_scheduler_func, ignore_cases))


def get_lr_scheduler_from_dict(optimizer, lr_scheduler_dict, ignore_cases=False):
    return get_lr_scheduler_from_args(optimizer, **lr_scheduler_dict, ignore_cases=ignore_cases)
