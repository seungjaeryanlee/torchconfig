import copy

import torch.optim as optim

from torchconfig.filter import filter_args


old_SGD = copy.deepcopy(optim.SGD)
optim.SGD = lambda *args, **kwargs: old_SGD(*args, **filter_args(kwargs, old_SGD))
