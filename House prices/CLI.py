from config import config
import pandas as pd
import itertools
import os

class CLI():
    def __init__(self):
        self.type_models = list(config.models.keys())

    def __call__(self):
        return self.forward()

    def all_in_one(self) -> tuple[pd.DataFrame, bool, bool, bool, bool, list]:
        setting = {'Train': False, 'Test': False, 'Submission': False, 'FE': False}
        selected_models = []
        all_model_phrase = 'all model'
        df = None
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f'you selected models: {selected_models}')
            print('-----setting-----')
            for num, (key, value) in enumerate(setting.items(), 1):
                print(f'{num}) {key} | {'ON' if value else 'OFF'}')
            models = pd.DataFrame({
            self.type_models[0]: pd.Series(config.models[self.type_models[0]]),
            self.type_models[1]: pd.Series(config.models[self.type_models[1]]),
            self.type_models[2]: pd.Series(config.models[self.type_models[2]])
        }).fillna('')
            print('=====m - select model=====q - quit=====')
            print(models)
            cmd = input(">> ").lower()
            if cmd in ['1', '2', '3', '4']:
                key = list(setting.keys())[int(cmd)-1]
                setting[key] = not setting[key]
            elif cmd == all_model_phrase:
                selected_models.extend(itertools.chain.from_iterable(config.models.values()))
            elif cmd == 'm':
                idx = input("Индекс модели column row: ")
                cell = [int(x) for x in idx.split()]
                selected_models.append(models.iloc[cell[1], cell[0]])
            elif cmd == 'q':
                if setting['Train'] or setting['Test']:
                    if config.path.train is '': raise ValueError('path not found')
                    df = pd.read_csv(config.path.train)
                if setting['Submission']:
                    if config.path.train is '': raise ValueError('path not found')
                    df = pd.read_csv(config.path.test)
                return (df, setting['Train'], setting['Test'], setting['Submission'], setting['FE'], selected_models)

    def forward(self) -> tuple[pd.DataFrame, bool, bool, bool, bool, list]:
        return self.all_in_one()