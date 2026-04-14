import pandas as pd
import numpy as np
from config import config
from work_with_data import work

prep = work()

df = prep.from_csv()
X_train, y = prep.forward(df = df, FE = False)
print(X_train.std())
print(y.std())