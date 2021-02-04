#导入计算模块，对模块大致功能注解：
#numpy支持矩阵运算
#pandas用来做数据清洗，像是python中的excel（支持将数据以.csv格式输出至本地）
#sklearn用来进一步制作数据集（支持数据的导入和数据的导出），含有SVM支持向量机、DT决策树、KNN近邻、LR逻辑回归等封装好的模型，支持对数据进行交叉验证以调参。
#mlxtend用来实现集成学习：bagging, boosting, stacking
#lightgbm内有boosting tree（相比xgboost，改进了生成节点的方式）
#xgboost内有boosting tree
#os用来读取文件
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

# record the time
print("START DATA PROCESSING", datetime.now())

# Remove the index
# train id and test id are useless

train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# Remove the outliners to avoid over fitting
train = train[train.GrLivArea < 4500]
# reset index
train.reset_index(drop=True, inplace=True)

##########对预测目标数值进行对数变换和特征矩阵对象的创建-【开始】#######
# log1p就是log(1+x)，用来对房价数据进行数据预处理，它的好处是转化后的数据更加服从高斯分布，有利于后续的分类结果。
# 需要注意，最后需要将预测出的平滑数据还原，而还原过程就是log1p的逆运算expm1。
train["SalePrice"] = np.log1p(train['SalePrice'])
#单独取出训练数据中的房价列信息，存入y对象。
y = train['SalePrice'].reset_index(drop=True)
#.reset_index(drop=True)方法：在原有的索引列重置索引，不再另外添加新列。有必要使用reset_index吗？有必要的，不这样做y将有两套index，作为df的y将有两列。
#沿着水平的方向寻找列名为'SalePrice'的列（们），把它们对应的列统统删掉。得到了单纯的特征矩阵，存入train_features对象中。
train_features = train.drop(['SalePrice'], axis=1)
#test本来就没有房价列，所以它本来就是单纯的特征矩阵。
test_features = test

##########对预测目标数值进行对数变换和特征矩阵对象的创建-【结束】#######

##合并训练数据特征矩阵与测试数据特征矩阵，以便统一进行特征处理-【开始】##
#将训练数据中的特征矩阵和测试数据中的特征矩阵合并（.concat[矩阵1,矩阵2]），并对合并后的矩阵index重新编号（.reset_index(drop=True)）。
features = pd.concat([train_features, test_features]).reset_index(drop=True)
#检查合并后的矩阵的维数，核查合并结果。
print("Merged feature set size:", features.shape)
##合并训练数据特征矩阵与测试数据特征矩阵，以便统一进行特征处理-【结束】##
#######################################################数据导入和特征提取-【结束】################################################################################

##############################################################特征处理-【开始】###################################################################################
#对于列名为'MSSubClass'、'YrSold'、'MoSold'的特征列，将列中的数据类型转化为string格式。
features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

###############################填充空值-【开始】##########################
#按照以下各个特征列的实际情况，依次处理各个特征列中的空值（.fillna()方法）
features['Functional'] = features['Functional'].fillna('Typ') #空值填充为str型数据'Typ'
features['Electrical'] = features['Electrical'].fillna("SBrkr") #空值填充为str型数据"SBrkr"
features['KitchenQual'] = features['KitchenQual'].fillna("TA") #空值填充为str型数据"TA"
features["PoolQC"] = features["PoolQC"].fillna("None") #空值填充为str型数据"None"
#对于列名为'Exterior1st'、'Exterior2nd'、'SaleType'的特征列，使用列中的众数填充空值。
#	1.先查找数据列中的众数：使用df.mode()[]方法
#	  解释：df.mode(0或1,0表示对列查找，1表示对行查找)[需要查找众数的df列的index（就是df中的第几列）]，将返回数据列中的众数
#	2.使用.fillna()方法进行填充
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

#对于列名为'GarageYrBlt', 'GarageArea', 'GarageCars'的特征列，使用0填充空值。
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    features[col] = features[col].fillna(0)
#对于列名为'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'的特征列，使用字符串'None'填充空值。
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('None')


#聚合函数（按某一列关键字分组）groupby，它的特点是：将返回与传入方法的矩阵维度相同的单个序列。
#transform是与groupby（pandas中最有用的操作之一）通常组合使用，它对传入方法的矩阵进行维度不变的变换。具体变换方法写在括号中，通常会使用匿名函数，对传入矩阵的所有元素进行操作。
#对于features矩阵，按照'MSSubClass'列中的元素分布进行分组，被分组的数据列是'MSZoning'列。feature.groupby(被作为索引的列的名称)[被分组的数据列的名称]
#features.groupby('MSSubClass')['MSZoning']后，得到的是一个以'MSSubClass'为索引，以'MSZoning'为数据列的矩阵。
#.transform()方法将对'MSZoning'数据列进行()内的变换，它将返回和传入矩阵同样维度的矩阵。
#括号内是匿名函数，将对传入矩阵中的空值进行填充，使用的填充元素是传入矩阵中的众数。
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
