from omegaconf import OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    'path': {
        'train': f'{BASE_DIR}/data/train.csv',
        'test': f'{BASE_DIR}/data/test.csv',
        'submission': f'{BASE_DIR}/data/submission.csv'
    },
    'models': {
        'Linermodel': ('linerreg', 'lassocv', 'ridgecv', 'elasticnetcv', "SVR"),
        'Treemodel': ('decisiontree', 'random_forest', 'catboost', 'xgboost', 'lightgbm'),
        'KNN': ('knn'),
        },
    'args': {
        'randomstate': 42,
        'kfold': {
            'folds': 5,
            'repeat': 3
        },
    'target': 'SalePrice'
        },
    'modelmode': ['train', 'test', 'submission']
}

config = OmegaConf.create(config)