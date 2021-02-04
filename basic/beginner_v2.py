import numpy as np  # linear algebra
import pandas as pd  #
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import os

print("Loaded pkg")

# load data and feature extraction

print(os.listdir("./fanblade/data_sets/beginner"))

train = pd.read_csv('./fanblade/data_sets/beginner/train.csv')
test = pd.read_csv('./fanblade/data_sets/beginner/test.csv')

print("Train data set size:", train.shape)
print("Test data set size:", test.shape)

print("EDA:")
quantitative = [f for f in train.columns if train.dtypes[f] != 'object'] # those are numbers
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
print("quantitative:", quantitative)
print("qualitative:", qualitative)
# record the time

print("check missing ")
print("START DATA PROCESSING", datetime.now())