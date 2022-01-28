# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:43:03 2021

@author: nguy0936
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
    return print(best_hyperparams)




##========Load data
lowd_conv1_1, lowd_embedding1, y1, X_hand1 = load_feature(mdir1) # set1
lowd_conv1_2, lowd_embedding2, y2, X_hand2 = load_feature(mdir2) # set2

c1 = np.concatenate((lowd_conv1_1, lowd_embedding1), axis=1)
c2 = np.concatenate((lowd_conv1_2, lowd_embedding2), axis=1)

df = np.concatenate((c1, c2), axis=0)
df_hand = np.concatenate((X_hand1, X_hand2), axis=0)
y = np.concatenate((y1, y2), axis=0)

##========Split data for tranning and testing
# split data into train and test sets (80% for training and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 0) # note X vs Xhand


#hyper_obj(X_train, y_train) # take time to run this!


##========= BUILT CLASSIFIER
# define optimal hyperparameters
params_hand = {"objective":"binary:logistic",
          'max_depth': 13,
          'learning_rate': 0.07,
          'gamma': 1.65,
          'min_child_weight': 50,
          'colsample_bytree': 0.7,
          'reg_alpha': 1.0,
          'reg_lambda': 0.64,
          'scale_pos_weight': 1,
          'n_estimators': 200} 


params_deep = {"objective":"binary:logistic",
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
X_train_deep, X_test_deep, y_train_deep, y_test_deep = train_test_split(df, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0) 
xgb_clf_deep = XGBClassifier(**params_deep)
xgb_clf_deep.fit(X_train_deep, y_train_deep ) # train with deep features


X_train_hand, X_test_hand, y_train_hand, y_test_hand = train_test_split(df_hand, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0) 
xgb_clf_hand = XGBClassifier(**params_hand)
xgb_clf_hand.fit(X_train_hand, y_train_hand ) # train with handcraft features


# deep feature performance
y_test_deep_pred = xgb_clf_deep.predict_proba(X_test_deep)
print( roc_auc_score(y_test_deep, y_test_deep_pred[:,1]) )
fpr, tpr, _ = roc_curve(y_test_deep, y_test_deep_pred[:,1])
plt.plot(fpr, tpr , linestyle='-', label='Deep feature') 

# handcraft feature performance
y_test_hand_pred = xgb_clf_hand.predict_proba(X_test_hand)
print( roc_auc_score(y_test_hand, y_test_hand_pred[:,1]) )
fpr, tpr, _ = roc_curve(y_test_hand, y_test_hand_pred[:,1])
plt.plot(fpr, tpr , linestyle='-', label='Handcraft feature') 

# Baseline IOA performance
print( roc_auc_score(y, df_hand[:,21] ) )
fpr, tpr, _ = roc_curve(y, df_hand[:,21] ) 
plt.plot(fpr, tpr , linestyle='-', label='IOA_baseline') 

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.savefig('AUC_compare.pdf')  
plt.show()


#np.savetxt("R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\R script\\result_Conv1_embedding1.csv", c1, delimiter=",")
