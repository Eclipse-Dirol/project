from omegaconf import OmegaConf
import os
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    'selectedmodels': ['lightgbm', 'lassocv'],
    'selectedensemble': ['stacking'],
    'train': True,
    'use_submit': True,
    'FE': True,
    'savemodel': False,
    'param_on': True,
    'optuna': {
        'on': False,
        'n_trials': 10,
        'loss': 'RMSE'  # его нельзя изменить, нужно лезть в код
    },
    'ensemble': {
                'on': True,
                'finalmodel': 'linerreg',
                'cv': 5,
            },
    'NN': {
        'on': False,
        'device': 'cuda',
        'activationlayer': 'ReLU',
        'name_loss_func': 'MSELoss',
        'name_opt_func': 'Adam',
        'epoch': 1000,
        'dropout': 0.1,
        'weight': f'{BASE_DIR}/data/models/mlp_weights.pth',
        'verbose': True,
        'batch': 20
    },
    'path': {
            'train': f'{BASE_DIR}/data/train.csv',
            'test': f'{BASE_DIR}/data/test.csv',
            'submission': f'{BASE_DIR}/data/submission',
            'save': f'{BASE_DIR}/data/models'
        },
    'models': {
            'Linermodel': ['linerreg', 'lassocv', 'ridgecv', 'elasticnet', 'SVR'],
            'Treemodel': ['decisiontree', 'random_forest', 'catboost', 'xgboost', 'lightgbm'],
            'KNN': ['knn'],
            'NN':['mlp'],
            'Ensemble': ['stacking', 'voiting'],
        },
    'args': {
            'randomstate': 42,
            'njobs': -1,
            'kfold': {
                'folds': 5,
                'repeat': 3,
            },
            'target': 'SalePrice',
        },
    'param': {
        'linerreg': {},
        'lassocv': {'max_iter': 10000},
        'ridgecv': {},
        'elasticnet': {},
        'SVR': {},
        'decisiontree':{},
        'random_forest': {},
        'catboost': {
            'iterations': 4000,
            'loss_function': 'RMSE',
            'bootstrap_type': 'MVS',
            'min_data_in_leaf': 10,
            'subsample': 0.8,
            'learning_rate': 0.004682365945633432,
            'depth': 8,
            'l2_leaf_reg': 1.8350145845846084,
            'random_strength': 7.4399645273858335,
            'verbose': False,
        },
        'xgboost': {
            'learning_rate': 0.03644265345990077,
            'max_depth': 3,
            'num_leaves': 34,
            'min_child_weight': 21,
            'reg_alpha': 0.5274555514454423,
            'reg_lambda': 0.06720022361481678,
            'gamma': 0.016632967950723738,
            'n_estimators': 4000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'verbosity': 0,
            'eval_metric': 'rmse',
            'n_jobs': -1
 },
        'lightgbm': {
            'learning_rate': 0.029162572866580794,
            'max_depth': 8,
            'num_leaves': 3,
            'min_child_samples': 2,
            'reg_alpha': 0.3137153336331008,
            'reg_lambda': 0.005369638446125523,
            'verbosity': 0,
            'n_estimators': 4000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1
        },
        'knn': {},
    },
    'optuna_param': {
        'elasticnet': {
            'alpha': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 1},
            'l1_ratio': {'type': 'float', 'log': False, 'low': 0, 'high': 1},
            'max_iter': {'type': 'int', 'log': False, 'low': 500, 'high': 3000},
            'selection': {'type': 'str', 'list': ['cyclic', 'random']},
        },
        'SVR': {
            'C': {'type': 'float', 'log': True, 'low': 1e-3, 'high': 100},
            'epsilon': {'type': 'float', 'log': False, 'low': 0, 'high': 1}
        },
        'decisiontree':{
            'max_depth': {'type': 'int', 'log': False, 'low': 5, 'high': 15},
            'min_samples_split': {'type': 'float', 'log': False, 'low': 0, 'high': 1},
            'min_samples_leaf': {'type': 'float', 'log': False, 'low': 0, 'high': 1},
            
        },
        'random_forest': {
            'n_estimators': {'type': 'int', 'log': False, 'low': 50, 'high': 400},
            'max_depth': {'type': 'int', 'log': False, 'low': 4, 'high': 15},
            'min_samples_split': {'type': 'float', 'log': False, 'low': 0, 'high': 1},
            'min_samples_leaf': {'type': 'float', 'log': False, 'low': 0, 'high': 1},
        },
        'catboost': {
            'iterations': {'type': 'int', 'log': False, 'low': 500, 'high': 3000},
            'learning_rate': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 1},
            'depth': {'type': 'int', 'log': False, 'low': 5, 'high': 13},
            'loss_function ': {'type': 'str', 'list': ['RMSE']},
            'l2_leaf_reg': {'type': 'int', 'log': False, 'low': 0, 'high': 25}
        },
        'xgboost': {
            'learning_rate': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 1},
            'min_split_loss': {'type': 'int', 'log': False, 'low': 0, 'high': 256},
            'max_depth': {'type': 'int', 'log': False, 'low': 5, 'high': 8},
            'min_child_weight': {'type': 'int', 'log': False, 'low': 0, 'high': 25},
            'subsample': {'type': 'float', 'log': False, 'low': 0.5, 'high': 1},
            'reg_lambda': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 10},
            'reg_alpha': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 10},
            'colsample_bytree': {'type': 'float', 'log': False, 'low': 0.5, 'high': 1},
            'n_estimators': {'type': 'int', 'log': False, 'low': 500, 'high': 3000}
        },
        'lightgbm': {
            'learning_rate': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 1},
            'n_estimators': {'type': 'int', 'log': False, 'low': 500, 'high': 3000},
            'num_leaves': {'type': 'int', 'log': False, 'low': 1, 'high': 5},
            'max_depth': {'type': 'int', 'log': False, 'low': 5, 'high': 13},
            'min_data_in_leaf': {'type': 'int', 'log': False, 'low': 5, 'high': 150},
            'feature_fraction': {'type': 'float', 'log': False, 'low': 0.5, 'high': 1},
            'bagging_fraction': {'type': 'float', 'log': False, 'low': 0.5, 'high': 1},
            'lambda_l1': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 10},
            'lambda_l2': {'type': 'float', 'log': True, 'low': 1e-4, 'high': 10},
        },
        'knn': {
            'n_neighbors': {'type': 'int', 'log': False, 'low': 4, 'high': 25},
            'leaf_size': {'type': 'int', 'log': False, 'low': 20, 'high': 150},
            'weights': {'type': 'str', 'list': ['uniform', 'distance']}
        },
    },
}

config = OmegaConf.create(config)

class Models():
    def __call__(self, name):
        return self.__get_func(name = name)

    def __get_func(self, name: str = None):
        dict_with_models = {'linerreg': LinearRegression, 'lassocv': LassoCV, 'ridgecv': RidgeCV, 'elasticnet': ElasticNet,
                            'SVR': LinearSVR, 'decisiontree': DecisionTreeRegressor,
                            'random_forest': RandomForestRegressor, 'catboost': CatBoostRegressor,
                            'xgboost': XGBRegressor, 'lightgbm': LGBMRegressor,
                            'knn': KNeighborsRegressor, 'stacking': StackingRegressor, 'voiting': VotingRegressor}
        return dict_with_models[name]