from omegaconf import OmegaConf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    'path': {
        'Hp': {
        'train': f'{BASE_DIR}/data/train.csv',
        'test': f'{BASE_DIR}/data/test.csv',
        'submission': f'{BASE_DIR}/data/submission.csv'
        }
    }
}

config = OmegaConf.create(config)