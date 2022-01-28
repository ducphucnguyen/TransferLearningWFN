# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:24:09 2021

@author: nguy0936
I used this code to classify noise at different location using Xhand data
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




##========== load low-high level features
mdir1 = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\Data set\\set1\\'
mdir2 = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\Data set\\set2\\'
mdir6 = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\Data set\\set6\\'
mdir9 = 'R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\Data set\\set9_WL8\\'


def load_feature(mdir): # function to load data

    conv1 = pd.read_csv(mdir + 'result_conv1.csv', header=None)   # conv1
    conv2 = pd.read_csv(mdir + 'result_conv2.csv', header=None)   # conv2
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
    
    return lowd_conv1, lowd_embedding, y, X_hand


lowd_conv1_1, lowd_embedding1, y1, X_hand1 = load_feature(mdir1) # set1
lowd_conv1_2, lowd_embedding2, y2, X_hand2 = load_feature(mdir2) # set2
lowd_conv1_6, lowd_embedding6, y6, X_hand6 = load_feature(mdir6) # set6
lowd_conv1_9, lowd_embedding9, y9, X_hand9 = load_feature(mdir9) # set9


df = np.concatenate((X_hand1, X_hand2, X_hand6), axis=0)
#df_hand = np.concatenate((X_hand1, X_hand2), axis=0)
#y_AM = np.concatenate((y1, y2), axis=0)

scaled_df = StandardScaler().fit_transform(df)

y1 = 1*np.ones((5000,), dtype=int)
y2 = 2*np.ones((3000,), dtype=int)
y6 = 3*np.ones((1000,), dtype=int)
#y9 = 4*np.ones((1000,), dtype=int)

y = np.concatenate((y1, y2, y6), axis=0)

# plot umap
reducer = umap.UMAP(random_state=42, 
                    n_neighbors=20,
                    min_dist=0.0,
                    metric='euclidean',
                    n_components=2)

data_umap = reducer.fit_transform(scaled_df)


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
    
    y_test = label_binarize(y_test, classes=[1, 2, 3])
    
    
    # print( roc_auc_score(y_test, y_test_pred) )
    
    return roc_auc_score(y_test, y_test_pred) 
    
    
#AUC = np.empty([10, 1])

#for i in range(0,10):

#    AUC[i] = clf_location(data_umap,y)





LALC = df[:, [13, 15]]
LALC[:,1] = np.multiply(LALC[:,0], LALC[:,1])

AUC2 = np.empty([10, 1])

for i in range(0,10):

    AUC2[i] = clf_location(LALC,y)
    
    
    
    
    
    