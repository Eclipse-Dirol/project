import pandas as pd
import numpy as np
from config import config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch

class work():
    def __init__(self):
        self.scaler  = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler.set_output(transform='pandas')

    def from_csv(self, train: bool = True) -> pd.DataFrame:
        if not isinstance(train, bool): raise ValueError('train not bool | work_with_data -> from_csv')
        if train:
            return pd.read_csv(config.path.train)
        return pd.read_csv(config.path.test)

    def lite_fillna(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame): raise ValueError('df isn`t pd.DataFrame | work_with_data -> lite_fillna')
        df_temp = df.copy()
        nan_test = df_temp.select_dtypes(include=['float', 'int']).isna().sum().index.tolist()
        for i in nan_test:
            df_temp[i] = df_temp[i].fillna(df_temp[i].median())
        df_temp = df_temp.fillna('unknown')
        return df_temp

    def transform_and_scaler(
        self,
        df: pd.DataFrame = None,
        use_submit: bool = False,
        nn: bool | None = None
    ) -> pd.DataFrame | torch.Tensor:
        if not isinstance(df, pd.DataFrame): raise ValueError('df isn`t pd.DataFrame | work_with_data -> transform_and_scaler')
        if not isinstance(use_submit, bool): raise ValueError('use_submit is not bool, fuck | work_with_data -> transform_and_scaler')
        df_temp = df.copy()
        if use_submit is False:
            num_cols = df_temp.select_dtypes(include=['float', 'int']).columns.tolist()
            cat_cols = df_temp.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
            ohe_df = self.encoder.fit_transform(df_temp[cat_cols])
            ohe_df = pd.DataFrame(ohe_df, columns=self.encoder.get_feature_names_out(cat_cols), index=df_temp.index)
            df_temp = df_temp.drop(columns=cat_cols)
            df_temp[num_cols] = self.scaler.fit_transform(df_temp[num_cols])
            df_temp = pd.concat([df_temp, ohe_df], axis=1)
            if nn:
                df_temp = df_temp.to_numpy()
                return torch.tensor(data = df_temp, dtype=torch.float32)
            return df_temp
        else:
            num_cols = df_temp.select_dtypes(include=['float', 'int']).columns.tolist()
            cat_cols = df_temp.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
            ohe_df = self.encoder.transform(df_temp[cat_cols])
            ohe_df = pd.DataFrame(ohe_df, columns=self.encoder.get_feature_names_out(cat_cols), index=df_temp.index)
            df_temp = df_temp.drop(columns=cat_cols)
            df_temp[num_cols] = self.scaler.transform(df_temp[num_cols])
            df_temp = pd.concat([df_temp, ohe_df], axis=1)
            if nn:
                df_temp = df_temp.to_numpy()
                return torch.tensor(data = df_temp, dtype=torch.float32)
            return df_temp

    def forward(
        self, 
        df: pd.DataFrame = None, 
        use_submit: bool = False, 
        FE: bool | None = None,
        nn: bool | None = None
    ) -> tuple[pd.DataFrame, pd.Series | None] | tuple[torch.Tensor, torch.Tensor | None, int]:
        if not isinstance(df, pd.DataFrame): raise ValueError('df don`t pd.DataFrame | work_with_data -> forward')
        if not isinstance(use_submit, bool): raise ValueError('use_submit is not bool, fuck | work_with_data -> forwar')
        
        df_temp = df.copy()
        if use_submit:
            df_temp = df_temp.drop(columns=['Id'])
        else:
            target = np.log1p(df_temp[config.args.target])
            df_temp = df_temp.drop(columns=[config.args.target, 'Id'])
        if FE:
            pass
        df_temp = self.lite_fillna(df = df_temp)
        df_temp = self.transform_and_scaler(df = df_temp, use_submit = use_submit, nn = nn)
        if use_submit:
            if nn:
                input_feat = df_temp.shape[1]
                return (df_temp, None, input_feat)
            return (df_temp, None)
        if nn:
            target = target.to_numpy()
            input_feat = df_temp.shape[1]
            return (df_temp, torch.tensor(target, dtype=torch.float32).view(-1, 1), input_feat)
        return (df_temp, target)