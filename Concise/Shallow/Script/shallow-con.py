import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import math
from math import sqrt

np.random.seed(1337)
train = pd.read_csv("./../Input/Train_Data-Con.csv")
test = pd.read_csv("./../Input/Test_Data-Con.csv")
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
    #class_labels = clf.classes_
    #y_pred_proba = clf.predict_proba(X)
    y_pred_proba_1 = y_pred[:,1]
    rms = sqrt(mean_squared_error(y_true, y_pred_proba_1))
    return rms
ftwo_scorer = make_scorer(my_scorer,  greater_is_better=False, needs_proba=True)

params = {
    'n_estimators': 480, 
    'num_leaves' : 70,
    'learning_rate': 0.03,
	'subsample': 0.88,
    'colsample_bytree': 0.65,
    'max_depth': -1,
    'min_child_samples':10,
    'max_bin' : 255,
    'nthread': 1
    
}

clf_xg = lgb.LGBMClassifier(**params)

clf_xg.fit(X, Y_concise)

test.Cat1 = test.Cat1.astype('category')
test.Cat2 = test.Cat2.astype('category')
test.Cat3 = test.Cat3.astype('category')
test.Country = test.Country.astype('category')
test.level = test.level.astype('category')

probs_con = clf_xg.predict_proba(test)
np.savetxt("./../OutPut/concise_test.predict", probs_con[:,1])
#print XGscore_con
#print XGgrid_search_con.best_params_
