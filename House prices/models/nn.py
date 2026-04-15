import torch
import torch.nn as nn
import pandas as pd
import torch
from config import config


class _model_nn(nn.Module):
    def __init__(
        self,
        input_feat,
        output_feat,
        dropout: float = config.NN.dropout,
    ):
        if input_feat is None: raise ValueError('input_feat in mlp is None')
        super().__init__()
        activation_class = getattr(nn, config.NN.activationlayer)
        self.activation = activation_class()
        self.layer_input = nn.Linear(input_feat, output_feat)
        self.batchnorm_input = nn.BatchNorm1d(output_feat)
        self.layer_out = nn.Linear(output_feat, 1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.batchnorm_input(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

class mlp():
    def __init__(
        self,
        device: str | None = None,
        input_feat: int | None = None,
        output_feat: int | None = None,
    ):
        if device is None:
            self.model = _model_nn(input_feat=input_feat, output_feat=output_feat)
        else:
            self.model = _model_nn(input_feat=input_feat, output_feat=output_feat).to(device)

    def predict(
        self,
        x: torch.Tensor = None,
        param_on: bool | None = None
    ):
        if param_on:
            self.model.load_state_dict(torch.load(config.NN.weight))
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def train(
        self,
        x: torch.Tensor =  None,
        y: torch.Tensor = None,
        name_loss_func: str = None,
        name_opt_func: str = None,
        EPOCH: int = None,
        verbose: bool = None,
        save_weight: bool = None,
    ):
        self.model.train()
        loss_fn = getattr(nn, name_loss_func)()
        opt = getattr(torch.optim, name_opt_func)(self.model.parameters(), lr=0.01)
        loss_list = []
        for epoch in range(EPOCH):
            opt.zero_grad()
            preds = self.model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            if verbose: 
                print(f'epoch {epoch+1}, loss: {loss}')
            loss_list.append(loss)
            if (len(loss_list)) > 2:
                if (loss_list[-2] - loss_list[-1]).abs() < 0.0005:
                    print(f'EARLY STOPPING HERE!!!')
                    break
        if save_weight:
            torch.save(self.model.state_dict(), config.NN.weight)