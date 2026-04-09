import pandas as pd
import numpy as np
from config import config
from preprocessing import work
from models.linear_models import linerreg, SVR, lassocv, ridgecv, elasticnetcv
from models.tree_models import dectree, random_forest, catboost, xgboost
from CLI import CLI

cli = CLI()
prep = work()

def main():
    output = {}
    choice_with_df, list_of_model = cli()
    df, train, test, use_submit, fe = choice_with_df
    print('===================start prep===================')
    df, target = prep.prep(
        df = df,
        train = train,
        test = test,
        FE = fe,
    )
    print('===================start train===================')
    if train:
        for name_model in list_of_model:
            model_class = globals().get(name_model)
            model = model_class()
            metrics = model.train(X_train = df, y = target, folds = config.args.kfold.folds, repeat = config.args.kfold.repeat)
            output[name_model] = metrics
    print(output['lassocv']['mae'][0])
    
    
if __name__ == '__main__':
    main()