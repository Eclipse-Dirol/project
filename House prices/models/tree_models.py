import numpy as np
import pandas as pd
from config import config
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from models.base import Base, validation
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

val = validation()

class dectree(Base):
    def __init__(self):
        self.model = DecisionTreeRegressor()

    def train(
        self,
        X_train: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        folds: int | None = None,
        repeat: int | None = None,
        train_size: float = 0.2
    ):
        if folds is None:
            metrics = val.none_folds(
                X = X_train,
                y = y,
                model = self.model,
                train_size=train_size
            )
            return metrics
        else:
            metrics = val.k_folds(
                X = X_train,
                y = y,
                folds=folds,
                repeat=repeat,
                model=self.model,
            )
            return metrics

    def predict(
        self,
        X: pd.DataFrame = None,
    ):
        return self.model.predict(X)

class random_forest(Base):
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(
        self,
        X_train: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        folds: int | None = None,
        repeat: int | None = None,
        train_size: float = 0.2
    ):
        if folds is None:
            metrics = val.none_folds(
                X = X_train,
                y = y,
                model = self.model,
                train_size=train_size
            )
            return metrics
        else:
            metrics = val.k_folds(
                X = X_train,
                y = y,
                folds=folds,
                repeat=repeat,
                model=self.model,
            )
            return metrics

    def predict(
        self,
        X: pd.DataFrame = None,
    ):
        return self.model.predict(X)

class catboost(Base):
    def __init__(self):
        self.model = CatBoostRegressor(verbose=False)

    def train(
        self,
        X_train: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        folds: int | None = None,
        repeat: int | None = None,
        train_size: float = 0.2
    ):
        if folds is None:
            metrics = val.none_folds(
                X = X_train,
                y = y,
                model = self.model,
                train_size=train_size
            )
            return metrics
        else:
            metrics = val.k_folds(
                X = X_train,
                y = y,
                folds=folds,
                repeat=repeat,
                model=self.model,
            )
            return metrics

    def predict(
        self,
        X: pd.DataFrame = None,
    ):
        return self.model.predict(X)

class xgboost(Base):
    def __init__(self):
        self.model = XGBRegressor(verbose=False)

    def train(
        self,
        X_train: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        folds: int | None = None,
        repeat: int | None = None,
        train_size: float = 0.2
    ):
        if folds is None:
            metrics = val.none_folds(
                X = X_train,
                y = y,
                model = self.model,
                train_size=train_size
            )
            return metrics
        else:
            metrics = val.k_folds(
                X = X_train,
                y = y,
                folds=folds,
                repeat=repeat,
                model=self.model,
            )
            return metrics

    def predict(
        self,
        X: pd.DataFrame = None,
    ):
        return self.model.predict(X)