import pandas as pd
import numpy as np
from config import config
from preprocessing import work
from models.linear_models import linerreg, SVR, lassocv, ridgecv, elasticnetcv
from models.tree_models import decisiontree, random_forest, catboost, xgboost, lightgbm
from models.KNN_models import knn
from CLI import CLI

cli = CLI()
prep = work()

def main():
    output = {}
    df, train, test, use_submit, fe, model_list = cli()
    print('===================start prep===================')
    df, target = prep.prep(
        df = df,
        train = train,
        test = test,
        FE = fe,
    )
    print('===================start train===================')
    if train:
        for name_model in model_list:
            model_class = globals().get(name_model)
            model = model_class()
            metrics = model.train(X_train = df, y = target, folds = config.args.kfold.folds, repeat = config.args.kfold.repeat)
            output[name_model] = metrics
    for model_name in output.keys():
        metrics_model = output[model_name]
        print(f'====={model_name}=====')
        print(f'MSE: {metrics_model['mse'][0]} | MSE_std: {metrics_model['mse'][0]}')
        print(f'MAE: {metrics_model['mae'][0]} | MAE_std: {metrics_model['mae'][0]}')
        print(f'R2: {metrics_model['r2']}')
    
    
if __name__ == '__main__':
    main()