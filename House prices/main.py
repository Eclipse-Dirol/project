import pandas as pd
import numpy as np
from config import config, Models
from work_with_data import work
from models.base import ModelPipeline
from sklearn.ensemble import StackingRegressor, VotingRegressor
import os

prep = work()
get_model = Models()
pipeline = ModelPipeline

def save_preds(preds: np.ndarray = None, idx: pd.Series = None, name: str = None):
    preds = pd.DataFrame({
                            'Id': idx,
                            config.args.target: preds
                        })
    preds.to_csv(f'{config.path.submission}/{name}.csv', index = False)

def get_metrics(metrics: dict = None):
    print(f'MSE: {metrics['mse'][0]} | MSE std: {metrics['mse'][1]}')
    print(f'MAE: {metrics['mae'][0]} | MAE std: {metrics['mae'][1]}')
    print(f'r2: {metrics['r2']}')

def main():
    use_submit = config.use_submit
    FE = config.FE
    save_model = config.savemodel
    ensemble = config.ensemble.on
    print(f'=+=+=+=+=+=+= start import and prep =+=+=+=+=+=+=')
    df = prep.from_csv()
    X_train, y = prep.forward(df = df, FE = FE)
    X_test = None
    if use_submit:
        df_test = prep.from_csv(train = False)
        idx = df_test['Id']
        X_test, _ = prep.forward(df = df_test, use_submit = use_submit, FE = FE)

    if ensemble is False:

        if config.train:

            print(f'=+=+=+=+=+=+= start train models =+=+=+=+=+=+=')
            models_name = config.selectedmodels

            for name in models_name:
                model = get_model(name = name)(**config.param[name])
                print(f'=+=+=+=+=+=+= {name} =+=+=+=+=+=+=')
                pipe = pipeline(model = model)
                metrics, preds = pipe.full_test(
                    X_train = X_train,
                    y = y,
                    folds = config.args.kfold.folds,
                    repeat = config.args.kfold.repeat,
                    use_submit = use_submit,
                    X_test = X_test
                )
                os.system('cls' if os.name == 'nt' else 'clear')
                get_metrics(metrics= metrics)
                if use_submit or save_model:
                    print('=+=+= save =+=+=')
                if use_submit:
                    save_preds(preds = preds, idx = idx, name = name)
                if save_model:
                    pipe.save_with_fit(
                        X_train = X_train,
                        y = y,
                        name = name
                    )

        else:

            print(f'=+=+=+=+=+=+= start train models =+=+=+=+=+=+=')
            models_name = config.selectedmodels
            for name in models_name:
                print(f'=+=+=+=+=+=+= {name} =+=+=+=+=+=+=')
                model = get_model(name = name)(**config.param[name])
                pipe = pipeline(model = model)
                pipe.fit(X = X_train, y = y)
                preds = pipe.predict( X = X_test)
                if use_submit or save_model:
                    print('=+=+= save =+=+=')
                if use_submit:
                    save_preds(preds = preds, idx = idx, name = name)
                if save_model:
                    pipe.save_with_fit(
                        X_train = X_train,
                        y = y,
                        name = name
                    )

    else:

        final_estimator = get_model(name = config.ensemble.finalmodel)()
        list_func_model = []
        for name in config.selectedmodels:
            list_func_model.append((name, get_model(name = name)(**config.param[name])))

        if config.train:

            print(f'=+=+=+=+=+=+= start train models =+=+=+=+=+=+=')
            for name in config.selectedensemble:
                ensemble = get_model(name = name)
                print(f'=+=+=+=+=+=+= {name} =+=+=+=+=+=+=')
                match name:
                    case 'stacking':
                        ensemble = ensemble(
                            estimators = list_func_model,
                            final_estimator = final_estimator,
                            cv = config.ensemble.cv,
                            n_jobs = config.args.njobs
                        )
                    case 'voiting':
                        ensemble = ensemble(
                            estimators = list_func_model,
                            n_jobs = config.args.njobs
                        )
                pipe = pipeline(model = ensemble)
                metrics, preds = pipe.full_test(
                    X_train= X_train,
                    y = y,
                    folds = config.args.kfold.folds,
                    repeat = config.args.kfold.repeat,
                    use_submit = use_submit,
                    X_test = X_test
                )
                get_metrics(metrics= metrics)
                if use_submit or save_model:
                    print('=+=+= save =+=+=')
                if use_submit:
                    save_preds(preds = preds, idx = idx, name = name)
                if save_model:
                    pipe.save_with_fit(
                        X_train = X_train,
                        y = y,
                        name = name
                    )

        else:

            print(f'=+=+=+=+=+=+= start train models =+=+=+=+=+=+=')
            for name in config.selectedensemble:
                print(f'=+=+=+=+=+=+= {name} =+=+=+=+=+=+=')
                ensemble = get_model(name = name)
                match name:
                    case 'stacking':
                        ensemble = ensemble(
                            estimators = list_func_model,
                            final_estimator = final_estimator,
                            cv = config.ensemble.cv,
                            n_jobs = config.args.njobs
                        )
                    case 'voiting':
                        ensemble = ensemble(
                            estimators = list_func_model,
                            n_jobs = config.args.njobs
                        )
                pipe = pipeline(model = ensemble)
                pipe.fit(X = X_train, y = y)
                preds = pipe.predict(X = X_test)
                if use_submit or save_model:
                    print('=+=+= save =+=+=')
                if use_submit:
                    save_preds(preds = preds, idx = idx, name = name)
                if save_model:
                    pipe.save_with_fit(
                        X_train = X_train,
                        y = y,
                        name = name
                    )

if __name__ == '__main__':
    main()