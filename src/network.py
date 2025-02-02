
from collections import OrderedDict
from torch import nn


class FFNetwork(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = OrderedDict()

        layers["lin0"] = nn.Linear(in_size, hidden_sizes[0])
        layers["relu0"] = nn.ReLU()
        layers["drop0"] = nn.Dropout(p=dropout)

        for i in range(1, len(hidden_sizes)):
            layers[f"lin{i}"] = nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            layers[f"relu{i}"] = nn.ReLU()
            layers[f"drop{i}"] = nn.Dropout(p=dropout)

        layers[f"lin{len(hidden_sizes)}"] = nn.Linear(hidden_sizes[-1], out_size)

        self.linear_relu_stack = nn.Sequential(layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
