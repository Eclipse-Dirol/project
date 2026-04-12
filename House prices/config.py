from omegaconf import OmegaConf
import os
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    'path': {
            'train': f'{BASE_DIR}/data/train.csv',
            'test': f'{BASE_DIR}/data/test.csv',
            'submission': f'{BASE_DIR}/data/submission',
            'save': f'{BASE_DIR}/data/models'
        },
    'models': {
            'Linermodel': ['linerreg', 'lassocv', 'ridgecv', 'elasticnetcv', 'SVR'],
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
            'NN': {
                'device': 'cuda',
                'layers': 4,
                'activationlayer': 'ReLU',
                'nums_layers': 4
                
            },
            'target': 'SalePrice',
        },
    'param': {
        'linerreg': {},
        'lassocv': {},
        'ridgecv': {},
        'elasticnetcv': {},
        'SVR': {},
        'decisiontree':{},
        'random_forest': {},
        'catboost': {'verbose': False},
        'xgboost': {'verbosity': 0},
        'lightgbm': {'verbose': -1},
        'knn': {},
    },
    'train': True,
    'use_submit': False,
    'FE': False,
    'savemodel': False,
    'selectedmodels': ['linerreg', 'lassocv', 'elasticnetcv', 'catboost', 'xgboost'],
    'selectedensemble': ['stacking'],
    'ensemble': {
                'on': True,
                'finalmodel': 'linerreg',
                'cv': 5,
            },
}

config = OmegaConf.create(config)

class Models():
    def __call__(self, name):
        return self.__get_func(name = name)

    def __get_func(self, name: str = None):
        dict_with_models = {'linerreg': LinearRegression, 'lassocv': LassoCV, 'ridgecv': RidgeCV, 'elasticnetcv': ElasticNetCV,
                            'SVR': LinearSVR, 'decisiontree': DecisionTreeRegressor,
                            'random_forest': RandomForestRegressor, 'catboost': CatBoostRegressor,
                            'xgboost': XGBRegressor, 'lightgbm': LGBMRegressor,
                            'knn': KNeighborsRegressor, 'mlp': None, 'stacking': StackingRegressor, 'voiting': VotingRegressor}
        return dict_with_models[name]