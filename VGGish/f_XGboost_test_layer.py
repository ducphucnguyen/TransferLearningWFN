# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:31:56 2021

@author: nguy0936

I increased the number of layers from conv1 to embedding to see if more layers
could result in better performance. I did this for only set 1 - Hallett
"""

# load packages
import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv
from sklearn.metrics import roc_auc_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


##========== load low-high level features
mdir1 = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\Data set\\set1\\'
mdir2 = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\Data set\\set2\\'

def load_feature(mdir): # function to load data

    conv1 = pd.read_csv(mdir + 'result_conv1.csv', header=None)   # conv1
    conv2 = pd.read_csv(mdir + 'result_conv2.csv', header=None)   # conv2
    conv3 = pd.read_csv(mdir + 'result_conv3.csv', header=None)   # conv3
    conv4 = pd.read_csv(mdir + 'result_conv4.csv', header=None)   # conv4
    embedding = pd.read_csv(mdir + 'result_embedding.csv', header=None)  # embedding
    X_hand = pd.read_csv(mdir + 'X_hand.csv')  # bias features
    X_hand = X_hand.fillna(0)
    
    Y = pd.read_csv(mdir + 'Y.csv', header=None)   # score
    y = Y
    y[:]=np.where(y<3,0,1)
    
    # combine data
    lowd_conv1 = PCA(n_components=10).fit_transform(conv1)
    lowd_conv2 = PCA(n_components=10).fit_transform(conv2)
    lowd_embedding = PCA(n_components=20).fit_transform(embedding)
    
    lowd_frames = [pd.DataFrame(lowd_conv1), pd.DataFrame(lowd_conv2), pd.DataFrame(lowd_embedding)]
    
    lowd_df = pd.concat(lowd_frames, axis=1)
    scaled_lowd_df = StandardScaler().fit_transform(lowd_df)
    
    return lowd_conv1, lowd_conv2, conv3, conv4, lowd_embedding, y


##=======Optimise hyperparamter
def hyper_obj(X_train, y_train):
    # define data_dmatrix, this is required to use XGboost API native
    dtrain = xgb.DMatrix(data=X_train, label=y_train)


    # DEFINE HYPERPARAMTER DISTRIBUTIONS
    # quniform : discrete uniform (integers spaced evenly)
    # uniform: continuous uniform (floats spaced evenly)
    
    space={'max_depth': hp.quniform("max_depth", 1, 20, 1),
           'learning_rate': hp.uniform('learning_rate', 0.05, 1),
            'gamma': hp.uniform ('gamma', 1, 20),
            'reg_alpha' : hp.quniform('reg_alpha', 0, 30, 1),
            'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.1, 1),
            'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1)
        }


    # DEFINE OBJECTIVE FUNCTION
    # general model of this function is: Input ->> Function >> Loss
    # Loss metric in here is area under the specificity-sensitivity curve (AUC)
    
    def objective(space):
        # max_depth should be in int data type
        space['max_depth'] = int(space['max_depth'])
    
        # run cross-validation
        cv_results = xgb.cv(
            space,
            dtrain,
            num_boost_round=500,
            seed=42,
            nfold=5,
            metrics={'aucpr'},
            early_stopping_rounds=10)
    
        # best AUC
        best_auc = cv_results['test-aucpr-mean'].max()
        loss = 1 - best_auc
        print(best_auc)
    
        # Dictionary with information for evaluation
        return {'loss': loss, 'params': space, 'status': STATUS_OK}


    # OPTIMISE THE OBJECTIVE FUNCTION
    # This is actually find the hyperparamters corresponding with lowest 1-AUC
    
    trials = Trials()
    
    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)
    
    # PRIN THE BEST HYPERPARAMTERS 
    print("The best hyperparameters are : ","\n")
    return best_hyperparams


## Performance evaluation
def eval_perform(df,y):
    
    # split data into train and test sets (80% for training and 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 0) # note X vs Xhand

    # evaluate performance on 20% testing data
    best_hyperparams = hyper_obj(X_train, y_train) # take time to run this!


    # define optimal hyperparameters
    params = {"objective":"binary:logistic",
              'max_depth': int(best_hyperparams['max_depth']),
              'learning_rate': best_hyperparams['learning_rate'],
              'gamma': best_hyperparams['gamma'],
              'min_child_weight': best_hyperparams['min_child_weight'],
              'colsample_bytree': best_hyperparams['colsample_bytree'],
              'reg_alpha': best_hyperparams['reg_alpha'],
              'reg_lambda': best_hyperparams['reg_lambda'],
              'scale_pos_weight': 1,
              'n_estimators': 200} 

    xgb_clf = XGBClassifier(**params)
    xgb_clf.fit(X_train, y_train) # train with deep features
    
    y_test_pred = xgb_clf.predict_proba(X_test)
    AUC = roc_auc_score(y_test, y_test_pred[:,1]) 
    
    return AUC


##========Load data
lowd_conv1_1, lowd_conv2_1, conv3_1, conv4_1, lowd_embedding_1, y_1 = load_feature(mdir1) # set1
lowd_conv1_2, lowd_conv2_2, conv3_2, conv4_2, lowd_embedding_2, y_2 = load_feature(mdir2) # set2


c1 = np.concatenate((lowd_conv1_1, lowd_embedding_1), axis=1)
c2 = np.concatenate((lowd_conv1_2, lowd_embedding_2), axis=1)

df = np.concatenate((c1, c2), axis=0)
y = np.concatenate((y_1, y_2), axis=0)

AUC = np.empty([10, 1])

for i in range(0,10):

    AUC[i] = eval_perform(df,y)


print(AUC)

print( np.mean(AUC) )
print( np.std(AUC) )



