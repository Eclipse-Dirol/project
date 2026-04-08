import numpy as np
import pandas as pd
from config import config
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from models.base_class import Base
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import is_regressor
from sklearn.model_selection import RepeatedKFold

class dectree(Base):
    def __init__(self):
        pass