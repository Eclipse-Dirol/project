from omegaconf import OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    'path': {
        'train': f'{BASE_DIR}/data/train.csv',
        'test': f'{BASE_DIR}/data/test.csv',
        'submission': f'{BASE_DIR}/data/submission.csv'
    },
    'model': {
        'linermodel': ('linerreg', 'LassoCV', 'RidgeCV', "SVR"),
        'treemodel': ('dectree', 'random_forest', 'catboost')
        },
    'args': {
        'randomstate': 42,
        'kfold': {
            'folds': 5,
            'repeat': 3
        },
        'target': 'SalePrice'
        },
    'metrics': ['mean_absolute_error', 'mean_squared_erroe', 'r2_score']    
}

config = OmegaConf.create(config)