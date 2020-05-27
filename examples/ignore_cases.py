import torch
import torchconfig


def main():
    # Sample configuration file
    CONFIG = {
        "seed": 1,
        "optimizer": {"name": "SGD", "lr": 0.1,},
        "lr_scheduler": {"name": "CyclicLR", "base_lr": 0.01, "max_lr": 1,},
        "activation": {"name": "Softshrink", "lambd": 0.8,},
    }

    # Initialize neural network, optimizer, and lr_scheduler
    params = [torch.FloatTensor([0])]
    optimizer = torchconfig.get_optimizer_from_args(params, name="SGD", lr=0.1, ignore_cases=True)
    optimizer = torchconfig.get_optimizer_from_args(params, name="SGD", lr=0.1, ignore_cases=False)
    optimizer = torchconfig.get_optimizer_from_args(params, name="SGD", lr=0.1)
    lr_scheduler = torchconfig.get_lr_scheduler_from_args(optimizer, **CONFIG["lr_scheduler"], ignore_cases=True)
    lr_scheduler = torchconfig.get_lr_scheduler_from_args(optimizer, **CONFIG["lr_scheduler"], ignore_cases=False)
    lr_scheduler = torchconfig.get_lr_scheduler_from_args(optimizer, **CONFIG["lr_scheduler"])
    lr_scheduler = torchconfig.get_lr_scheduler_from_args(optimizer, NAME="CyclicLR", BASE_LR=0.01, MAX_LR=1, ignore_cases=True)
    lr_scheduler = torchconfig.get_lr_scheduler_from_dict(optimizer, CONFIG["lr_scheduler"], ignore_cases=True)
    lr_scheduler = torchconfig.get_lr_scheduler_from_dict(optimizer, CONFIG["lr_scheduler"], ignore_cases=False)
    lr_scheduler = torchconfig.get_lr_scheduler_from_dict(optimizer, CONFIG["lr_scheduler"])


if __name__ == "__main__":
    main()
