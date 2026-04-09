from config import config
import pandas as pd
import itertools

class CLI():
    def __init__(self):
        self.model_mode = list(config.modelmode)
        self.type_models = list(config.models.keys())

    def __call__(self):
        return self.forward()

    def choice_train_test_val(self) -> tuple[pd.DataFrame, bool, bool, bool, bool]:
        df = None
        use_train = use_test = use_submit = fe = False
        answer = int(input(f'Choice what you want: \n 1) {self.model_mode[0]} \n 2) {self.model_mode[1]}(if model have weight) \n 3) {self.model_mode[2]}(if model have weight) \n you answer: '))
        if answer == 1:
            if config.path.train == '': raise ValueError('path not found')
            df = pd.read_csv(config.path.train)
            fe_answer = int(input('Do you want to do FE?\n 1) Yes\n 2) No\nyou choose:'))
            if fe_answer == 1:
                fe = True
            use_train = True
        elif answer == 2:
            if config.path.train == '': raise ValueError('path not found')
            df = pd.read_csv(config.path.train)
            fe_answer = int(input('Do you want to do FE?\n 1) Yes\n 2) No\nyou choose:'))
            if fe_answer == 1:
                fe = True
            use_test = True
        elif answer == 3:
            if config.path.test == '': raise ValueError('path not found')
            df = pd.read_csv(config.path.test)
            use_submit = True
        return (df, use_train, use_test, use_submit, fe)

    def choice_model(self) -> list:
        models = pd.DataFrame({
            self.type_models[0]: pd.Series(config.models[self.type_models[0]]),
            self.type_models[1]: pd.Series(config.models[self.type_models[1]])
        }).fillna('')
        print('====================================================================')
        print(models)
        print('Enter your answer in the format: column row\n(If you want to exit, enter: "exit", if you want to add all models, enter: "all model")')
        selected_models = []
        exit_phrase = 'exit'
        all_model_phrase = 'all model'
        while True:
            input_ = input()
            if input_ == all_model_phrase:
                return list(itertools.chain.from_iterable(config.models.values()))
            if input_ == exit_phrase:
                return selected_models
            cell = [int(x) for x in input_.split()]
            column, row = cell[0], cell[1]
            selected_models.append(models.iloc[row, column])
            print(f'you selected models: {selected_models}')

    def forward(self) -> tuple[tuple[pd.DataFrame, bool, bool, bool, bool], list]:
        df, train, test, use_submit, fe = self.choice_train_test_val()
        list_of_models = self.choice_model()
        output = (df, train, test, use_submit, fe)
        return (output, list_of_models)