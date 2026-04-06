import pandas as pd
import numpy as np
from config import config
from work_with_models import work

work = work()
df_train = pd.read_csv(config.path.Hp.train)
print(df_train.shape)
df_train = work.prep(df=df_train)
print(df_train.isna().sum().sum())