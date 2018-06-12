#primary objective to predict the value of SalePrice - advanced regression model
#import some required librairies
import numpy as np # for the purpose of linear algebra
import pandas as pd # data processing
from pandas import read_csv # to read the file
import matplotlib.pyplot as plt  # plotting/visualization
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warning(*args, **kwargs):
    pass
warnings.warn = ignore_warning # To ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew #for some foundational statistics
#check the files available in the directory
from subprocess import check_output
print(check_output(["ls", "//Users/house prices dataset"]).decode("utf8")
#STEP 1 - Get the feel of the dataset
#Let us read the train and test datasets
train = read_csv("train.csv")
test = read_csv("test.csv")
#display the first few rows of the train dataset.
print(train.head(10))
print(test.head(10))
print(train.columns)
print(test.columns)
print(train.shape)
print(test.shape)
#descriptive statistics summary for target variable - sale price
print(train['SalePrice'].describe())
#visualize the sale price using histogram
sns.distplot(train['SalePrice'])
plt.show()
#output from the viz we see - Deviation from the normal distribution,Have appreciable positive skewness,Shows peakedness.
#skewness and kurtosis :
#Skewness measures the extent to which the data values are not symmetrical around the mean.
#Kurtosis affects the peakedness of the curve of the distribution—that is, how sharply the curve rises approaching the center of the distribution.
print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())
#STEP2 - data cleaning and preprocessing
#outliers viz plot
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#remove the outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
print(len(train))
print(train.count())
print(train['SalePrice'].count())
#to see the log transformation - in  a two way table form
df = pd.concat([train.SalePrice, np.log(train.SalePrice + 1).rename('LogSalePrice')], axis=1, names=['SalePrice', 'LogSalePrice'])
#transformation using viz
plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice, kde=False, fit = norm)
#log plot
plt.subplot(1, 2, 2)
sns.distplot(np.log(train.SalePrice + 1), kde=False, fit = norm)
plt.xlabel('Log SalePrice')
plt.show()
#Get also the QQ-plot - viz before transformation
fig = plt.figure()
res = stats.probplot(train.SalePrice, plot=plt)
plt.show()
#Get also the QQ-plot - viz after transformation to reduce the skewness
fig = plt.figure()
res = stats.probplot(np.log(train.SalePrice + 1), plot=plt)
plt.show()
#let us save the ID column for any analysis purpose if needed
train_id = train['Id']
test_id = test['Id']
#Now drop the  'Id' column since it's unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
#concatenate the train and test data in the same dataframe
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
#figure out NAs
nulls = np.sum(all_data.isnull())
nullcols = nulls.loc[(nulls != 0)]
print(nullcols.sort_values(ascending = False))
print("There are", len(nullcols), "columns with missing values")
#Correlation map to see how features are correlated with SalePrice
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
#Handling missing values
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
# shape
#print('Shape all_data: {}'.format(all_data.shape))
# Adding total sqfootage feature
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
#print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
#print(skewness.head(10))
skewness = skewness[abs(skewness) > 0.75]
#print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data = pd.get_dummies(all_data)
#print(train.shape,test.shape)
#print(all_data.shape)
#print(all_data.columns)
train = all_data[:ntrain]
test = all_data[ntrain:]
#print(train.shape,test.shape)
