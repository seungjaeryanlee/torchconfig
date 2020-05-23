import copy

import torch.optim as optim

from torchconfig.filter import filter_args


old_SGD = copy.deepcopy(optim.SGD)
optim.SGD = lambda *args, **kwargs: old_SGD(*args, **filter_args(kwargs, old_SGD))

old_ReduceLROnPlateau = copy.deepcopy(optim.lr_scheduler.ReduceLROnPlateau)
optim.lr_scheduler.ReduceLROnPlateau = lambda *args, **kwargs: old_ReduceLROnPlateau(*args, **filter_args(kwargs, old_ReduceLROnPlateau))
