import torch
import torch.nn as nn
import pandas as pd

from config import config


class model_nn(nn.Module):
    def __init__(
        self,
        input_feat,
        out_feat
    ):
        if input_feat is None: raise ValueError('input_feat in mlp is None')
        super().__init__()
        activation_class = getattr(nn, config.args.NN.activationlayer)
        self.activation = activation_class()
        self.layer_input = nn.Linear(input_feat, out_feat)
        self.layer_out = nn.Linear(out_feat, 1)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.activation(x)
        x = self.layer_out(x)
        return x

class mlp():
    def __init__(
        self,
        device: str | None = None,
        input_feat: int | None = None,
        out_feat: int | None = None,
    ):
        if device is None:
            self.model = model_nn(input_feat=input_feat, out_feat=out_feat)
        else:
            self.model = model_nn(input_feat=input_feat, out_feat=out_feat).to(device)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def train(
        self,
        x,
        y,
        name_loss_func: str = None,
        name_opt_func: str = None,
        EPOCH: int = None,
    ):
        self.model.train()
        loss_fn = getattr(nn, name_loss_func)
        opt = getattr(torch.optim, name_opt_func)
        for epoch in EPOCH:
            opt.zero_grad()
            preds = self.model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()