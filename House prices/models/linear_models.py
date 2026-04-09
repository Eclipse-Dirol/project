import numpy as np
import pandas as pd
from config import config
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import LinearSVR
from models.base import Base, validation

val = validation()

class linerreg(Base):
    def __init__(self):
        self.model = LinearRegression()

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

class lassocv(Base):
    def __init__(self):
        self.model = LassoCV()

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

class ridgecv(Base):
    def __init__(self):
        self.model = RidgeCV()

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

class elasticnetcv(Base):
    def __init__(self):
        self.model = ElasticNetCV()

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

class SVR(Base):
    def __init__(self):
        self.model = LinearSVR()

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
