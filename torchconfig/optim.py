import copy

import torch.optim as optim

from torchconfig.filter import filter_args


# Optimizers
old_ASGD = copy.deepcopy(optim.ASGD)
optim.ASGD = lambda *args, **kwargs: old_ASGD(*args, **filter_args(kwargs, old_ASGD))

old_Adadelta = copy.deepcopy(optim.Adadelta)
optim.Adadelta = lambda *args, **kwargs: old_Adadelta(*args, **filter_args(kwargs, old_Adadelta))

old_Adagrad = copy.deepcopy(optim.Adagrad)
optim.Adagrad = lambda *args, **kwargs: old_Adagrad(*args, **filter_args(kwargs, old_Adagrad))

old_Adam = copy.deepcopy(optim.Adam)
optim.Adam = lambda *args, **kwargs: old_Adam(*args, **filter_args(kwargs, old_Adam))

old_AdamW = copy.deepcopy(optim.AdamW)
optim.AdamW = lambda *args, **kwargs: old_AdamW(*args, **filter_args(kwargs, old_AdamW))

old_Adamax = copy.deepcopy(optim.Adamax)
optim.Adamax = lambda *args, **kwargs: old_Adamax(*args, **filter_args(kwargs, old_Adamax))

old_LBFGS = copy.deepcopy(optim.LBFGS)
optim.LBFGS = lambda *args, **kwargs: old_LBFGS(*args, **filter_args(kwargs, old_LBFGS))

old_RMSprop = copy.deepcopy(optim.RMSprop)
optim.RMSprop = lambda *args, **kwargs: old_RMSprop(*args, **filter_args(kwargs, old_RMSprop))

old_Rprop = copy.deepcopy(optim.Rprop)
optim.Rprop = lambda *args, **kwargs: old_Rprop(*args, **filter_args(kwargs, old_Rprop))

old_SGD = copy.deepcopy(optim.SGD)
optim.SGD = lambda *args, **kwargs: old_SGD(*args, **filter_args(kwargs, old_SGD))

# Learning Rate Schedulers
old_CosineAnnealingLR = copy.deepcopy(optim.lr_scheduler.CosineAnnealingLR)
optim.lr_scheduler.CosineAnnealingLR = lambda *args, **kwargs: old_CosineAnnealingLR(*args, **filter_args(kwargs, old_CosineAnnealingLR))

old_CosineAnnealingWarmRestarts = copy.deepcopy(optim.lr_scheduler.CosineAnnealingWarmRestarts)
optim.lr_scheduler.CosineAnnealingWarmRestarts = lambda *args, **kwargs: old_CosineAnnealingWarmRestarts(*args, **filter_args(kwargs, old_CosineAnnealingWarmRestarts))

old_CyclicLR = copy.deepcopy(optim.lr_scheduler.CyclicLR)
optim.lr_scheduler.CyclicLR = lambda *args, **kwargs: old_CyclicLR(*args, **filter_args(kwargs, old_CyclicLR))

old_ExponentialLR = copy.deepcopy(optim.lr_scheduler.ExponentialLR)
optim.lr_scheduler.ExponentialLR = lambda *args, **kwargs: old_ExponentialLR(*args, **filter_args(kwargs, old_ExponentialLR))

old_LambdaLR = copy.deepcopy(optim.lr_scheduler.LambdaLR)
optim.lr_scheduler.LambdaLR = lambda *args, **kwargs: old_LambdaLR(*args, **filter_args(kwargs, old_LambdaLR))

old_MultiStepLR = copy.deepcopy(optim.lr_scheduler.MultiStepLR)
optim.lr_scheduler.MultiStepLR = lambda *args, **kwargs: old_MultiStepLR(*args, **filter_args(kwargs, old_MultiStepLR))

old_MultiplicativeLR = copy.deepcopy(optim.lr_scheduler.MultiplicativeLR)
optim.lr_scheduler.MultiplicativeLR = lambda *args, **kwargs: old_MultiplicativeLR(*args, **filter_args(kwargs, old_MultiplicativeLR))

old_OneCycleLR = copy.deepcopy(optim.lr_scheduler.OneCycleLR)
optim.lr_scheduler.OneCycleLR = lambda *args, **kwargs: old_OneCycleLR(*args, **filter_args(kwargs, old_OneCycleLR))

old_ReduceLROnPlateau = copy.deepcopy(optim.lr_scheduler.ReduceLROnPlateau)
optim.lr_scheduler.ReduceLROnPlateau = lambda *args, **kwargs: old_ReduceLROnPlateau(*args, **filter_args(kwargs, old_ReduceLROnPlateau))

old_StepLR = copy.deepcopy(optim.lr_scheduler.StepLR)
optim.lr_scheduler.StepLR = lambda *args, **kwargs: old_StepLR(*args, **filter_args(kwargs, old_StepLR))
