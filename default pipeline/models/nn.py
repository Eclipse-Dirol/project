import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from config import config
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

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
        self.sigmoid = nn.Sigmoid()
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
        x = self.sigmoid(x)

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

    def create_train_loader(
        self,
        X: torch.Tensor = None,
        y: torch.Tensor = None,
        batch_size: int = None
    ):
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader

    def train(
        self,
        name_loss_func: str = None,
        name_opt_func: str = None,
        EPOCH: int = None,
        verbose: bool = None,
        save_weight: bool = None,
        train_loader = None
    ):
        self.model.train()
        loss_fn = getattr(nn, name_loss_func)()
        opt = getattr(torch.optim, name_opt_func, torch.optim.Adam)(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=config.NN.epoch, optimizer=opt, eta_min=1e-5)
        loss_list = []
        for epoch in range(EPOCH):
            epoch_loss = 0
            acc=0
            for batch in train_loader:
                X_batch, y_batch = batch
                X_batch = X_batch.to(config.NN.device)
                y_batch = y_batch.to(config.NN.device)
                opt.zero_grad()
                preds = self.model(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                acc += accuracy_score(y_batch.detach().cpu().numpy(), 
                                      (preds.detach().cpu().numpy() > 0.5).astype(int))
            scheduler.step()
            if verbose:
                epoch_loss /= len(train_loader)
                acc /= len(train_loader)
                print(f'epoch {epoch+1}, loss: {epoch_loss},  ACC: {acc}')
            loss_list.append(loss)
            if (len(loss_list)) > 2:
                if (loss_list[-2] - loss_list[-1]).abs() < 0.001:
                    print(f'EARLY STOPPING HERE!!! with loss: {loss}')
                    print(f'ACC: {accuracy_score(y_batch.detach().cpu().numpy(), 
                                                (preds.detach().cpu().numpy() > 0.5).astype(int))}')
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
        param_on: bool = None,
        idx: pd.Series | None = None,
    ):
        if train:
            train_loader = self.create_train_loader(
                X = X_train,
                y = y,
                batch_size=config.NN.batch
            )
            self.train(
                name_loss_func=config.NN.name_loss_func,
                name_opt_func=config.NN.name_opt_func,
                EPOCH=config.NN.epoch,
                verbose=config.NN.verbose,
                save_weight=save_weight,
                train_loader=train_loader
            )
        if use_submit:
            preds = self.predict(x = X_test, param_on=param_on).to('cpu').view(-1)
            binary_preds = (preds >= 0.5).to(torch.int)
            preds = pd.DataFrame({
                            'PassengerId': idx,
                            config.args.target: binary_preds
                        })
            preds.to_csv(f'{config.path.submission}/mlp.csv', index = False)