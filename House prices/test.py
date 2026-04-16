import pandas as pd
import numpy as np
from config import config
from work_with_data import work
from models.nn import mlp

prep = work()

df = prep.from_csv()
df_test = prep.from_csv(train = False)
X_train, y = prep.forward(df = df, FE = False, nn = True)
X_test, _ = prep.forward(df = df_test, FE = False, nn = True, use_submit=True)
idx = df_test['Id']
input_feat = X_train.shape[1]
model = mlp(device = 'cpu', input_feat=input_feat, output_feat=input_feat//2)
model.train(x = X_train, 
            y = y, 
            name_loss_func=config.NN.name_loss_func, 
            name_opt_func=config.NN.name_opt_func,
            EPOCH = 3000,
            verbose=True,
            save_weight = config.NN.weight,
        )
# preds = model.predict(x = X_test, param_on=True)
# preds = pd.DataFrame({
#                     'Id': idx,
#                     config.args.target: np.exp(preds.numpy().flatten())
#                 })
# preds.to_csv(f'{config.path.submission}/mlp.csv', index = False)