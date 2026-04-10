from sklearn.neighbors import KNeighborsRegressor
from models.base import Base, validation
import numpy as np
import pandas as pd
from config import config

val = validation()

class knn(Base):
    def __init__(self):
        self.model = KNeighborsRegressor()

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