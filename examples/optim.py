import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchconfig


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    # Sample configuration file
    CONFIG = {
        "seed": 1,
        "optimizer": {
            "name": "SGD",
            "lr": 0.1,
        },
        "lr_scheduler": {
            "name": "CyclicLR",
            "base_lr": 0.01,
            "max_lr": 1,
        },
    }

    # Synthetic dataset
    dataset_size = 10
    input_size = 2
    output_size = 1
    X = torch.randn(dataset_size, input_size)
    Y = torch.randn(dataset_size, output_size)

    # Initialize neural network, optimizer, and lr_scheduler
    net = Net(input_size=2, output_size=1)
    optimizer = torchconfig.get_optimizer_from_dict(net.parameters(), CONFIG["optimizer"])
    lr_scheduler = torchconfig.get_lr_scheduler_from_dict(optimizer, CONFIG["lr_scheduler"])

    # One epoch of training
    Y_hat = net(X)
    loss = F.mse_loss(Y, Y_hat)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lr_scheduler.step()


if __name__ == "__main__":
    main()
