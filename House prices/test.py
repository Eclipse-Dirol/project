import pandas as pd
import numpy as np
from config import config
from models.linear_models import linerreg, SVR, lassocv, ridgecv
from work_with_models import work

prep = work()
model_1 = linerreg()
model_2 = SVR()
model_3 = lassocv()
model_4 = ridgecv()
print('--import data--')
df_train = pd.read_csv(config.path.train)
print('--import complete, prep now--')
X, y = prep.prep(df = df_train)
print('--prep ends, start train--')
metrics_1 = model_1.train(X_train = X, y = y, folds = 5, repeat = 3)
metrics_2 = model_2.train(X_train = X, y = y, folds = 5, repeat = 3)
metrics_3 = model_3.train(X_train = X, y = y, folds = 5, repeat = 3)
metrics_4 = model_4.train(X_train = X, y = y, folds = 5, repeat = 3)
print('--linerreg:')
print(f'MSE: {metrics_1['mse'][0]} | MSE std: {metrics_1['mse'][1]}')
print(f'MAE: {metrics_1['mae'][0]} | MAE std: {metrics_1['mae'][1]}')
print(f'r2: {metrics_1['r2']}')
print('--SVR:')
print(f'MSE: {metrics_2['mse'][0]} | MSE std: {metrics_2['mse'][1]}')
print(f'MAE: {metrics_2['mae'][0]} | MAE std: {metrics_2['mae'][1]}')
print(f'r2: {metrics_2['r2']}')
print('--lassocv:')
print(f'MSE: {metrics_3['mse'][0]} | MSE std: {metrics_3['mse'][1]}')
print(f'MAE: {metrics_3['mae'][0]} | MAE std: {metrics_3['mae'][1]}')
print(f'r2: {metrics_3['r2']}')
print('--ridgecv:')
print(f'MSE: {metrics_4['mse'][0]} | MSE std: {metrics_4['mse'][1]}')
print(f'MAE: {metrics_4['mae'][0]} | MAE std: {metrics_4['mae'][1]}')
print(f'r2: {metrics_4['r2']}')

# cols_with_nans = df_train.columns[df_train.isna().any()].tolist()

# # 2. Очищаем данные
# df_train = prep.prep_for_nan(df = df_train)

# # 3. Строим отчет только по тем колонкам, которые мы запомнили
# df_report = pd.DataFrame({
#     'dtype': df_train[cols_with_nans].dtypes,
#     'unique_values': df_train[cols_with_nans].nunique()
# }, index=cols_with_nans)

# print(df_report)