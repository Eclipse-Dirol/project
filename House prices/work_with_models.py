import pandas as pd
import numpy as np
from config import config
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class work():
    def __init__(self):
        pass
    
    def prep_for_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        func for nan processing
        nan -> 'unknown' for category features
        If you need to do FE, do it right away
        """
        df_temp = df.copy()
        df_temp['LotFrontage'] = df_temp['LotFrontage'].fillna(value=df_temp['LotFrontage'].median())
        df_temp['MasVnrArea'] = df_temp['MasVnrArea'].fillna(df_temp['MasVnrArea'].median())
        df_temp['GarageYrBlt'] = df_temp['GarageYrBlt'].fillna(0)
        df_temp = df_temp.fillna('unknown')
        return df_temp

    def prep_with_fe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FE and prep for df
        """
        df_temp = df.copy
        df_temp['LotFrontage'] = df_temp['LotFrontage'].fillna(value=df_temp['LotFrontage'].median())
        df_temp['MasVnrArea'] = df_temp['MasVnrArea'].fillna(df_temp['MasVnrArea'].median())
        df_temp['Have_Garage'] = (df_temp['GarageArea'] > 0).astype(int)
        df_temp['Have_second_floor'] = (df_temp['2ndFlrSF'] > 0).astype(int)
        df_temp['Old_House'] = ((df_temp['YearRemodAdd'] == df_temp['YearBuilt']) & (df_temp['YearBuilt'] <= df_temp['YearBuilt'].median())).astype(int)
        df_temp['Have_Pool'] = (df_temp['PoolArea'] > 0).astype(int)
        df_temp['High_Price'] = (df_temp['SalePrice'] > df_temp['SalePrice'].quantile(0.9)).astype(int)
        df_temp.drop(colomns=['GarageYrBlt'])

    def prep(
        self,
        df: pd.DataFrame = None, 
        train: bool = True,
        FE: bool = False
    ) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
        df_temp = df.copy()
        if train:
            target = df_temp[config.args.target]
            df_temp = df_temp.drop(columns=['Id', config.args.target])
            num_cols = df_temp.select_dtypes(include=['int', 'float']).columns.tolist()
            cat_cols = df_temp.select_dtypes(include=['str', 'category', 'object']).columns.tolist()
            col_trans = ColumnTransformer(transformers=[
                ('num_cols', StandardScaler(), num_cols),
                ('cat_cols', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
            ])
            col_trans.set_output(transform="pandas")
            pipe = Pipeline([
                ('prep_for_nan', FunctionTransformer(self.prep_for_nan)),
                ('prep', col_trans)
            ])
            df_temp = pipe.fit_transform(X=df_temp)
            return (df_temp, target)
        else:
            df_temp = self.prep_for_nan(df = df_temp)
            df_temp = df_temp.drop(columns=['Id'])
            return df_temp