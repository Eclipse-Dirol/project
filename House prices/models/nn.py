import torch
import torch.nn as nn
import pandas as pd
import torch
from config import config


class _model_nn(nn.Module):
    def __init__(
        self,
        input_feat: int = None,
    ):
        if input_feat is None: raise ValueError('input_feat in mlp is None')
        super().__init__()
        dropout: float = config.NN.dropout
        activation_class = getattr(nn, config.NN.activationlayer)
        self.activation = activation_class()
        self.layer_input = nn.Linear(input_feat, 256)
        self.batchnorm_input = nn.BatchNorm1d(256)
        self.hide_layer = nn.Linear(256, 128)
        self.batchnorm_hide = nn.BatchNorm1d(128)
        self.layer_out = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.batchnorm_input(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hide_layer(x)
        x = self.batchnorm_hide(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

class MLP():
    def __init__(
        self,
        input_feat: int = None,
    ):
        device = config.NN.device
        if device is None:
            self.model = _model_nn(input_feat=input_feat)
        else:
            self.model = _model_nn(input_feat=input_feat).to(device)

    def predict(
        self,
        x: torch.Tensor = None,
        param_on: bool | None = None
    ):
        x = x.to(config.NN.device)
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
        x = x.to(config.NN.device)
        y = y.to(config.NN.device)
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
                    print(f'EARLY STOPPING HERE!!! with loss: {loss}')
                    break
        if save_weight:
            torch.save(self.model.state_dict(), config.NN.weight)

    def forward(
        self,
        train: bool | None = None,
        use_submit: bool | None = None,
        X_train: torch.Tensor = None,
        y: torch.Tensor = None,
        X_test: torch.Tensor = None,
        save_weight: bool = None,
        param_on: bool = None
    ):
        if train:
            self.train(
                x = X_train,
                y = y,
                name_loss_func=config.NN.name_loss_func,
                name_opt_func=config.NN.name_opt_func,
                EPOCH=config.NN.epoch,
                verbose=config.NN.verbose,
                save_weight=save_weight
            )
        if use_submit:
            self.predict(x = X_test, param_on=param_on)