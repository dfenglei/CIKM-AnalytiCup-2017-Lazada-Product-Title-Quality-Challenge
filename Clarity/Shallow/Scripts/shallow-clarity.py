import pandas as pd
import numpy as np
import lightgbm as lgb
import math
from math import sqrt

np.random.seed(1337)
train = pd.read_csv("./../Input/Train_Data_CLR.csv")
test = pd.read_csv("./../Input/Test_Data_CLR.csv")
train.Cat1 = train.Cat1.astype('category')
train.Cat2 = train.Cat2.astype('category')
train.Cat3 = train.Cat3.astype('category')
train.Country = train.Country.astype('category')
train.level = train.level.astype('category')
#print train.columns

train = train.iloc[np.random.permutation(len(train))]
print train.index
Y_clarity = np.asarray(train[['clarity']].copy())
Y_concise = np.asarray(train[['concise']].copy())
Y_clarity = Y_clarity.reshape(Y_clarity.shape[0])
Y_concise = Y_concise.reshape(Y_concise.shape[0])
train.drop('clarity', axis=1, inplace=True)
train.drop('concise', axis=1, inplace=True)

print train.shape
print test.shape
X = train
print train.columns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

def my_scorer(y_true, y_pred):
    y_pred_proba_1 = y_pred[:,1]
    rms = sqrt(mean_squared_error(y_true, y_pred_proba_1))
    return rms
ftwo_scorer = make_scorer(my_scorer,  greater_is_better=False, needs_proba=True)

params = {'colsample_bytree': 0.6,
 'learning_rate': 0.03,
 'max_bin': 255,
 'max_depth': 6,
 'min_child_samples': 10,
 'n_estimators': 390,
 'num_leaves': 50,
 'subsample': 0.88,
 'nthread': 1}

clf_xg = lgb.LGBMClassifier(**params)

clf_xg.fit(X, Y_clarity)

test.Cat1 = test.Cat1.astype('category')
test.Cat2 = test.Cat2.astype('category')
test.Cat3 = test.Cat3.astype('category')
test.Country = test.Country.astype('category')
test.level = test.level.astype('category')

probs_clarity = clf_xg.predict_proba(test)
np.savetxt("./../Output/clarity_test.csv", probs_clarity[:,1])

