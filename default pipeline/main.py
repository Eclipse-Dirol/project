import pandas as pd
import numpy as np
from config import config, Models
from work_with_data import work
from models.base import ModelPipeline
from models.nn import MLP

prep = work()
get_model = Models()
pipeline = ModelPipeline
if config.NN.on:
    mlp = MLP

def save_preds(preds: np.ndarray = None, idx: pd.Series = None, name: str = None):
    binary_preds = (preds >= 0.5).astype(int)
    preds = pd.DataFrame({
                            'PassengerId': idx,
                            config.args.target: binary_preds
                        })
    preds.to_csv(f'{config.path.submission}/{name}.csv', index = False)

def get_metrics(metrics: dict = None):
    print(f'acc:  {metrics['acc'][0]}  | acc std:  {metrics['acc'][1]}')
    print(f'f1:  {metrics['f1'][0]}  | f1 std:  {metrics['f1'][1]}')
    print(f'roc_auc: {metrics['roc'][0]} | roc_auc std: {metrics['roc'][1]}')
    print(f'pr_auc: {metrics['pr'][0]} | pr_auc std: {metrics['pr'][1]}')

def main():
    use_submit = config.use_submit
    FE = config.FE
    save_model = config.savemodel
    ensemble = config.ensemble.on
    print(f'=+=+=+=+=+=+= start import and prep =+=+=+=+=+=+=')
    df = prep.from_csv()
    X_train, y = prep.forward(df = df, FE = FE)
    if config.NN.on:
        X_train_nn, y_nn, input_feat = prep.forward(df = df, FE = FE, nn = config.NN.on)
    X_test = None
    X_test_nn = None
    idx = None
    if use_submit:
        df_test = prep.from_csv(train = False)
        idx = df_test['PassengerId']
        if config.NN.on:
            X_test_nn, _, input_feat = prep.forward(df = df_test, FE = FE, nn = config.NN.on, use_submit=use_submit)
        X_test, _ = prep.forward(df = df_test, use_submit = use_submit, FE = FE)

    if ensemble is False:

        print(f'=+=+=+=+=+=+= start train models =+=+=+=+=+=+=')
        models_name = config.selectedmodels

        for name in models_name:
            if config.param_on and name != 'mlp':
                model = get_model(name = name)(**config.param[name])
            elif name == 'mlp':
                pass
            else:
                model = get_model(name = name)()
            print(f'=+=+=+=+=+=+= {name} =+=+=+=+=+=+=')
            if config.train:
                if name == 'mlp':
                    model = mlp(input_feat=input_feat)
                    model.forward(
                        train=config.train,
                        use_submit=use_submit,
                        X_train = X_train_nn,
                        y = y_nn,
                        X_test = X_test_nn,
                        save_weight=config.savemodel,
                        param_on = config.param_on,
                        idx=idx
                    )
                    pass
                else:
                    pipe = pipeline(model = model)
                    if config.optuna.on:
                        pipe.search_hyperparam_with_optuna(
                            n_trials = config.optuna.n_trials,
                            name = name,
                            X_train = X_train,
                            y = y,
                        )
                    else:
                        metrics, preds = pipe.full_test(
                            X_train = X_train,
                            y = y,
                            folds = config.args.kfold.folds,
                            use_submit = use_submit,
                            X_test = X_test
                        )
                        get_metrics(metrics= metrics)
            else:
                if name == 'mlp':
                    model = mlp(input_feat=input_feat)
                    model.forward(
                        train=config.train,
                        use_submit=use_submit,
                        X_train = X_train_nn,
                        y = y_nn,
                        X_test = X_test_nn,
                        save_weight=config.savemodel,
                        param_on = config.param_on,
                        idx=idx
                    )
                    continue
                else:
                    pipe = pipeline(model = model)
                    pipe.fit(X = X_train, y = y)
                    preds = pipe.predict( X = X_test)
            if use_submit or save_model:
                print('=+=+= save =+=+=')
            if use_submit and name != 'mlp':
                save_preds(preds = preds, idx = idx, name = name)
            if save_model and name != 'mlp':
                pipe.save_with_fit(
                    X_train = X_train,
                    y = y,
                    name = name
                )
        return 0

    else:

        final_estimator = get_model(name = config.ensemble.finalmodel)()
        list_func_model = []
        for name in config.selectedmodels:
            if config.param_on and name != 'mlp':
                model = get_model(name = name)(**config.param[name])
            elif name == 'mlp':
                pass
            else:
                model = get_model(name = name)()
            list_func_model.append((name, model))

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
                            n_jobs = config.args.njobs,
                            voting = 'soft'
                        )

                pipe = pipeline(model = ensemble)
                if config.train:
                    metrics, preds = pipe.full_test(
                        X_train= X_train,
                        y = y,
                        folds = config.args.kfold.folds,
                        use_submit = use_submit,
                        X_test = X_test
                    )
                    get_metrics(metrics= metrics)
                else:
                    pipe.fit(X = X_train, y = y)
                    preds = pipe.predict(X = X_test)
                if use_submit or save_model:
                    print('=+=+= save =+=+=')
                if use_submit and name != 'mlp':
                    save_preds(preds = preds, idx = idx, name = name)
                if save_model and name != 'mlp':
                    pipe.save_with_fit(
                        X_train = X_train,
                        y = y,
                        name = name
                    )
        return 0

if __name__ == '__main__':
    main()