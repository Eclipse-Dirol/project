import pandas as pd
import numpy as np
from config import config
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class work():
    def __init__(self):
        pass
    
    def lite_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def fe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        FE (feature engineering) for df
        """
        df_temp = df.copy
        target = df_temp[config.args.target]
        df_temp['LotFrontage'] = df_temp['LotFrontage'].fillna(value=df_temp['LotFrontage'].median())
        df_temp['MasVnrArea'] = df_temp['MasVnrArea'].fillna(df_temp['MasVnrArea'].median())
        df_temp['Have_Garage'] = (df_temp['GarageArea'] > 0).astype(int)
        df_temp['Have_second_floor'] = (df_temp['2ndFlrSF'] > 0).astype(int)
        df_temp['Old_House'] = ((df_temp['YearRemodAdd'] == df_temp['YearBuilt']) & (df_temp['YearBuilt'] <= df_temp['YearBuilt'].median())).astype(int)
        df_temp['Have_Pool'] = (df_temp['PoolArea'] > 0).astype(int)
        df_temp['High_Price'] = (df_temp['SalePrice'] > df_temp['SalePrice'].quantile(0.9)).astype(int)
        df_temp.drop(colomns=['GarageYrBlt'])
        return (df_temp, target)

    def transform_and_scaler(
        self,
        df: pd.DataFrame = None,
        train: bool | None = None,
        test: bool | None = None
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        df_temp = df.copy()
        if train or test:
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
            ('lite_fiilna', FunctionTransformer(self.lite_fillna)),
            ('prep', col_trans)
        ])
        df_temp = pipe.fit_transform(X=df_temp)
        if train or test:
            return (df_temp, target)
        return (df_temp, None)

    def prep(
        self,
        df: pd.DataFrame = None, 
        train: bool | None = None,
        test: bool | None = None,
        FE: bool = False
    ) -> tuple[pd.DataFrame, pd.Series]:
        df_temp = df.copy()
        if FE:
            df_temp, target = self.fe()
        else:
            df_temp, target = self.transform_and_scaler(df = df_temp, train = train, test = test)
            return (df_temp, target)