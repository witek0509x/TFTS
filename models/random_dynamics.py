import torch
import torch.nn as nn


class RandomDynamics(nn.Module):
    def __init__(self, d_model, lags):
        super(RandomDynamics, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(lags, d_model),
            nn.Sigmoid(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
            nn.Linear(d_model, 1))
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.reset()

    def forward(self, x):
        # Add learnable positional encoding
        # print("Random dynamics input", x)
        return self.network(x)

    def reset(self):
        nn.init.uniform_(self.network[0].weight, -10, 10)
        nn.init.uniform_(self.network[0].bias, -1, 1)
        nn.init.uniform_(self.network[2].weight, -10, 10)
        nn.init.uniform_(self.network[2].bias, -1, 1)
        nn.init.uniform_(self.network[4].weight, -10, 10)
        nn.init.uniform_(self.network[4].bias, -1, 1)
