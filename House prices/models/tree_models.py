import numpy as np
import pandas as pd
from config import config
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from models.base_class import Base
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import is_regressor
from sklearn.model_selection import RepeatedKFold
from catboost import CatBoostRegressor

def none_folds(
    X,
    y,
    train_size,
    model: any = None,
    random_state: int | None = config.args.randomstate
    ) -> dict:
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X not a DataFrame')
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError('y not s DataFrame or Series')
    assert 0 < train_size < 1, 'train_size must be between 0 and 1'
    if not is_regressor(model): raise ValueError('model not defined')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
        )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {'mae': mae, 'mse': mse, 'r2': r2}

def have_folds(
    X,
    y,
    folds,
    repeat,
    model: any = None,
    random_state: int | None = config.args.randomstate
    ) -> dict[tuple]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X not a DataFrame')
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError('y not s DataFrame or Series')
    assert folds>0 and repeat>0, 'only positive values for folds, repeat'
    if not is_regressor(model): raise ValueError('model not defined')
    
    kf = RepeatedKFold(n_splits=folds, n_repeats=repeat, random_state=random_state)
    mae_list, mse_list, r2_list = [], [], []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        
        mae_list.append(mean_absolute_error(y_val, preds))
        mse_list.append(mean_squared_error(y_val, preds))
        r2_list.append(r2_score(y_val, preds))

    mae, std_mae = np.mean(mae_list), np.std(mae_list)
    mse, std_mse = np.mean(mse_list), np.std(mse_list)
    r2 = np.mean(r2_list)
    
    return {'mae': (mae, std_mae), 'mse': (mse, std_mse), 'r2': (r2)}

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
            metrics = none_folds(
                X = X_train,
                y = y,
                model = self.model,
                train_size=train_size
            )
            return metrics
        else:
            metrics = have_folds(
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
            metrics = none_folds(
                X = X_train,
                y = y,
                model = self.model,
                train_size=train_size
            )
            return metrics
        else:
            metrics = have_folds(
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
            metrics = none_folds(
                X = X_train,
                y = y,
                model = self.model,
                train_size=train_size
            )
            return metrics
        else:
            metrics = have_folds(
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