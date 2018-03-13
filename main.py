import os

import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score

categorical_features = ['MSSubClass', 'MSZoning', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                        'Foundation', 'Heating', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
ordinal_features = ['LotShape', 'LandContour', 'Utilities', 'LandSlope',
                    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
numerical_features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                      'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                      'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces',
                      'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                      '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond']
bin_features = ['Street', 'CentralAir']
error = ['Bedroom', 'Kitchen']


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def write_to_submission_file(predicted_labels, out_file,
                             target='SalePrice', index_label="Id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1461, 2920),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

def find_seed(X, y, estim = 35):
    max_seed = 0
    max_score = -1;
    cur_score=0
    for seed in range(10, 50):
        cur_score = score(X, y, estim=estim, seed=seed)
        if (cur_score>max_score):
            max_seed = seed
            max_score=cur_score
    print("Max score: ", max_score)
    print("Seed: ", max_seed)
    return max_seed

from sklearn.ensemble import ExtraTreesRegressor
def score(X, y, estim = 10, seed = 17, ratio=0.9):
    idx = int(round(X.shape[0] * ratio))
    rf = ExtraTreesRegressor(n_estimators=estim, random_state=seed)
    rf.fit(X[:idx, :], y[:idx])
    y_pred = rf.predict(X[idx:, :])
    score = explained_variance_score(y_true=y[idx:], y_pred=y_pred)
    return score

def cmp(feature, values):
    for i in values:
        print("value:", i)
        print("max:", train_df[train_df[feature] == i]['SalePrice'].max())
        print("min:", train_df[train_df[feature] == i]['SalePrice'].min())
        print("mean:", train_df[train_df[feature] == i]['SalePrice'].mean())

def remove_noise(df):
    list_removed = []
    percentage = df.shape[0]*0.9
    for name in df.columns.values.tolist():
        y = df[name].value_counts().tolist()
        if (y[0]>=percentage):
            list_removed.append(name)
    df.drop(list_removed, axis=1)
    return list_removed

def replace_ordinal(df):
    df[ordinal_features] = df[ordinal_features].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], [5, 4, 3, 2, 1, 0])
    df['LotShape'] = df['LotShape'].map({'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0})
    df['PavedDrive'] = df['PavedDrive'].map({'Y': 2, 'N': 0, 'P': 1})
    df['Fence'] = df['Fence'].map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0})
    df['Functional'] = df['Functional'].map({'Typ': 0, 'Min1': -1, 'Min2': -2,
                                                         'Mod': -3, 'Maj1': -4, 'Maj2': -5, 'Sev': -6, 'Sal': -7})
    df['BsmtFinType1'] = df['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1})
    df['BsmtFinType2'] = df['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1})
    df['LandContour'] = df['LandContour'].map({'Lvl': 3, 'Bnk': 2, 'HLS': 1, 'Low': 0})
    df['LandSlope']=df['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})
    df['Utilities']=df['Utilities'].map({'AllPub': 3, 'NoSewr':2, 'NoSeWa':1, 'ELO':0})
    df['BsmtExposure']=df['BsmtExposure'].replace(['Av', 'Mn', 'No'], [3, 2, 1])
    df['GarageFinish'] = df['GarageFinish'].map({ 'Fin': 4, 'RFn':3, 'Rough':2, 'Unf':1, 'NA':0})
    df['Electrical'] = df['Electrical'].map({'Sbrkr': 2, 'FuseA': 2, 'FuseB': 1, 'FuseP': 1, 'Mix': 0})

def replace_categorical(df):
    tmp=df[categorical_features].apply(LabelEncoder().fit_transform)
    return OneHotEncoder().fit_transform(tmp).toarray()

def preprocessing(df):
    df = DataFrameImputer().fit_transform(df)
    df['Street'] = df['Street'].map({'Grvl': 0, 'Pave': 1})
    df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})
    replace_ordinal(df)
    X=np.hstack([df[bin_features],
                       df[numerical_features],
                       df[ordinal_features],
                       replace_categorical(df)])
    print(X)
    return X.astype('int')

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
y_train = train_df['SalePrice'].astype('int').values

full_df = pd.concat([train_df.drop(['SalePrice'], axis=1), test_df])
idx_split = train_df.shape[0]

full_df = preprocessing(full_df)
full_df = StandardScaler().fit_transform(full_df)


#print(len(categorical_features)+len(ordinal_features)+ len(numerical_features) + len(bin_features))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

estimator = GradientBoostingRegressor()
model = RFECV(estimator, step=1, cv=5)
model.fit(full_df[:idx_split, :], y_train)
full_df = model.transform(full_df)

# clf = Pipeline([
#    ('feature_selection', RFECV(ExtraTreesRegressor())),
#    ('classification', ExtraTreesRegressor())
#  ])


X_train = full_df[:idx_split, :]
print(X_train.shape)
s = find_seed(X_train, y_train, 64)
print(score(X_train, y_train, seed = s, estim=64))


clf = GradientBoostingRegressor(n_estimators=64, random_state = s)
clf.fit(X_train, y_train)
y_test = clf.predict(full_df[idx_split:, :])
print(y_test)
write_to_submission_file(y_test, 'submission.csv')