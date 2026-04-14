from config import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, root_mean_squared_log_error
from sklearn.base import is_regressor
from sklearn.model_selection import RepeatedKFold
from sklearn.base import BaseEstimator, RegressorMixin
import optuna
import joblib

class validation():
    @staticmethod
    def none_folds(
    X_train: pd.DataFrame = None,
    y: pd.Series = None,
    X_test: pd.DataFrame | None = None,
    train_size: float | None = config.args.randomstate,
    model: any = None,
    random_state: int | None = config.args.randomstate,
    use_submit: bool | None = None,
    param_on: bool = False,
    param: dict | None = None,
    ) -> tuple[dict, np.ndarray | None]:

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError('X not a DataFrame')
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError('y not s DataFrame or Series')
        assert 0 < train_size < 1, 'train_size must be between 0 and 1'
        if not is_regressor(model): raise ValueError('model not defined')
        if param_on:
            model = model(**param)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y, train_size=train_size, random_state=random_state
            )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_preds = None
        if use_submit:
            test_preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return ({'mae': mae, 'mse': mse, 'r2': r2}, test_preds)

    @staticmethod
    def k_folds(
        X_train: pd.DataFrame = None,
        y: pd.Series = None,
        X_test: pd.DataFrame | None = None,
        folds: int | None = config.args.kfold.folds,
        repeat: int | None = config.args.kfold.repeat,
        model: any = None,
        random_state: int | None = config.args.randomstate,
        use_submit: bool | None = None,
        param_on: bool = False,
        param: dict | None = None,
        ) -> tuple[dict, np.ndarray | None] | float:

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError('X not a DataFrame')
        if not isinstance(y, pd.Series):
            raise TypeError('y not a Series')
        assert folds>0 and repeat>0, 'only positive values for folds, repeat'
        if not is_regressor(model): raise ValueError('model not defined')
        if param_on:
            model = model(**param)
        kf = RepeatedKFold(n_splits=folds, n_repeats=repeat, random_state=random_state)
        mae_list, mse_list, rmse_list, rmsle_list, r2_list = [], [], [], [], []
        if use_submit:
            test_list = []
        else:
            test_preds = None
        for (tr_idx, val_idx) in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)

            if use_submit:
                if X_test is None: raise ValueError('X_test is None, please fix it')
                test_list.append(model.predict(X_test))
            mae_list.append(mean_absolute_error(y_val, preds))
            mse_list.append(mean_squared_error(y_val, preds))
            rmse_list.append(root_mean_squared_error(y_val, preds))
            rmsle_list.append(root_mean_squared_log_error(y_val, preds))
            r2_list.append(r2_score(y_val, preds))

        if use_submit:
            test_preds = np.mean(test_list, axis=0)
        mae, std_mae = np.mean(mae_list), np.std(mae_list)
        mse, std_mse = np.mean(mse_list), np.std(mse_list)
        rmse, std_rmse = np.mean(rmse_list), np.std(rmse_list)
        rmsle, std_rmsle = np.mean(rmsle_list), np.std(rmsle_list)
        r2 = np.mean(r2_list)
        return ({'mae': (mae, std_mae), 'mse': (mse, std_mse), 'rmse': (rmse, std_rmse), 'rmsle': (rmsle, std_rmsle), 'r2': (r2)}, test_preds)

    @staticmethod
    def k_folds_for_optuna(
        trial: int = None,
        X_train: pd.DataFrame = None,
        y: pd.Series = None,
        name: str = None,
        model: any = None,
        ):
        param = config.optuna_param[name]
        name_param = param.keys()
        for name in name_param:
            if param[name]['type'] == 'int':
                trial.suggest_int(name, param[name]['low'], param[name]['high'], log=param[name]['log'])
            elif param[name]['type'] == 'float':
                trial.suggest_float(name, param[name]['low'], param[name]['high'], log=param[name]['log'])
            elif param[name]['type'] == 'str':
                trial.suggest_categorical(name, param[name]['list'])
        model = model.set_params(**trial.params)
        kf = RepeatedKFold(n_splits=config.args.kfold.folds,
                           n_repeats=config.args.kfold.repeat,
                           random_state=config.args.randomstate)
        loss = []
        for (tr_idx, val_idx) in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            loss.append(root_mean_squared_error(y_val, preds))
        return np.mean(loss)
            

class ModelPipeline(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model: any = None,
    ):
        self.model = model
        self.val = validation()

    def fit(
        self,
        X: pd.DataFrame  = None,
        y: pd.Series = None,
    ):
        return self.model.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame = None,
    ):
        return self.model.predict(X)

    def full_test(
        self,
        X_train: pd.DataFrame = None,
        y: pd.Series = None,
        use_submit: bool | None = None,
        X_test: pd.DataFrame | None = None,
        param_on: bool | None = None,
        param: dict | None = None,
        folds: int | None = config.args.kfold.folds,
        optuna: bool | None = config.optuna.on,
    ) -> tuple[dict, np.ndarray | None]:
        if optuna:
            return self.val.k_folds_for_optuna(
                X_train = X_train,
                y= y,
                model = self.model,
                
            )
        if folds is None:
            metrics, preds = self.val.none_folds(
                X_train = X_train,
                y = y,
                model = self.model,
                X_test = X_test,
                use_submit = use_submit,
                param_on = param_on,
                param = param,
            )
            return (metrics, preds)
        else:
            metrics, preds = self.val.k_folds(
                X_train = X_train,
                y = y,
                model = self.model,
                X_test = X_test,
                use_submit = use_submit,
                param_on = param_on,
                param = param,
            )
            return (metrics, preds)

    def save_model(
        self,
        name: str = None,
    ):
        joblib.dump(self.model, f'{config.path.save}/{name}.pkl')
        return self

    def load_model(
        self,
        name: str = None,
    ) -> any:
        return joblib.load(f'{config.path.load}/{name}.pkl')

    def save_with_fit(
        self,
        X_train: pd.DataFrame = None,
        y: pd.Series = None,
        name: str = None
    ):
        self.fit(
                X = X_train,
                y = y
            )
        self.save_model(name = name)

    def search_hyperparam_with_optuna(
        self,
        n_trials: int = None,
        name: str = None,
        X_train: pd.DataFrame = None,
        y: pd.Series = None,
    ):
        study = optuna.create_study(direction='minimize')
        study.optimize(func = lambda trial: self.val.k_folds_for_optuna(trial, X_train, y, name, self.model), n_trials = n_trials)
        best_param = study.best_params
        print(f'best cv: {study.best_value}')
        print(best_param)