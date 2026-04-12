import torch
import torch.nn as nn
import pandas as pd

from config import config


class model(nn.Module):
    def __init__(
        self,
        input_feat: int | None = None
    ):
        if input_feat is None: raise ValueError('input_feat in mlp is None')
        super().__init__()
        activation_class = getattr(nn, config.args.NN.activationlayer)
        self.activation = activation_class()
        self.linear_layer_1 = nn.Linear(input_feat)

    def forward(self):
        pass

class mlp():
    def __init__(self):
        pass