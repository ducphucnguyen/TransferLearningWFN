# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 10:49:56 2021

@author: nguy0936

I used XGboost to classify noise measured at 4 locations
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
from sklearn.preprocessing import label_binarize


mdir1 = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\R script\\'

df = pd.read_csv( mdir1 + 'UMAP_spatial.csv', header=None)  

y1 = 1*np.ones((5000,), dtype=int)
y2 = 2*np.ones((3000,), dtype=int)
y6 = 3*np.ones((1000,), dtype=int)
y9 = 4*np.ones((1000,), dtype=int)

y = np.concatenate((y1, y2, y6, y9), axis=0)

#y = label_binarize(y, classes=[1, 2, 3, 4])



def clf_location(df,y):
    #========Split data for tranning and testing
    # split data into train and test sets (80% for training and 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2) # note X vs Xhand
    
    params_deep = {"objective":"multi:softmax",
              'max_depth': 19,
              'learning_rate': 0.13,
              'gamma': 1.11,
              'min_child_weight': 31,
              'colsample_bytree': 0.92,
              'reg_alpha': 5.0,
              'reg_lambda': 0.796,
              'scale_pos_weight': 1,
              'n_estimators': 200} 
    
    
    # train the classifier to the training data
    xgb_clf = XGBClassifier(**params_deep)
    xgb_clf.fit(X_train, y_train ) # train with deep features
    
    
    y_test_pred = xgb_clf.predict_proba(X_test)
    
    y_test = label_binarize(y_test, classes=[1, 2, 3, 4])
    
    
    # print( roc_auc_score(y_test, y_test_pred) )
    
    return roc_auc_score(y_test, y_test_pred) 
    
    
AUC = np.empty([10, 1])

for i in range(0,10):

    AUC[i] = clf_location(df,y)










