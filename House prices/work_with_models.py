import pandas as pd
import numpy as np
import torch.nn as nn
import torch

class work():
    def __init__(self):
        pass
    
    def prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """func for nan processing"""
        df_temp = df.copy()
        df_temp['LotFrontage'] = df_temp['LotFrontage'].fillna(value=df_temp['LotFrontage'].median())
        df_temp['MasVnrArea'] = df_temp['MasVnrArea'].fillna(df_temp['MasVnrArea'].median())
        df_temp = df_temp.fillna('unknown')
        return df_temp