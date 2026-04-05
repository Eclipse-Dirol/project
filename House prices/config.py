from omegaconf import OmegaConf

config = {
    'path': {
        'train': '../House prices/data/train.csv',
        'test': '../House prices/data/test.csv',
        'submission': '../House prices/data/submission.csv'
    }
    
}

config = OmegaConf.create(config)