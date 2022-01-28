# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:22:23 2021
This umap analysis is for spatial characteristics.
and for temporal characteristics 

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


c1 = np.concatenate((lowd_conv1_1, lowd_embedding1), axis=1)
c2 = np.concatenate((lowd_conv1_2, lowd_embedding2), axis=1)
c6 = np.concatenate((lowd_conv1_6, lowd_embedding6), axis=1)
c9 = np.concatenate((lowd_conv1_9, lowd_embedding9), axis=1)

df = np.concatenate((c1, c2, c6, c9), axis=0)
#df_hand = np.concatenate((X_hand1, X_hand2), axis=0)
#y_AM = np.concatenate((y1, y2), axis=0)

scaled_df = StandardScaler().fit_transform(df)

y1 = 1*np.ones((5000,), dtype=int)
y2 = 2*np.ones((3000,), dtype=int)
y6 = 3*np.ones((1000,), dtype=int)
y9 = 4*np.ones((1000,), dtype=int)

y_loc = np.concatenate((y1, y2, y6, y9), axis=0)


# plot umap
reducer = umap.UMAP(random_state=42, 
                    n_neighbors=20,
                    min_dist=0.0,
                    metric='euclidean',
                    n_components=2)

data_umap = reducer.fit_transform(scaled_df)

plt.scatter(data_umap[:,0], data_umap[:,1],
            c= y_loc,
            s=0.2,
            cmap='viridis')     

#plt.savefig('UMAP_spatial.pdf')  
plt.show()
#np.savetxt("R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\R script\\UMAP_spatial.csv", data_umap, delimiter=",")





###====================TEMPORAL CHARACTERISTICS

def load_feature_full(mdir): # function to load data

    conv1 = pd.read_csv(mdir + 'result_conv1_full.csv', header=None)   # conv1
    embedding = pd.read_csv(mdir + 'result_embedding_full.csv', header=None)  # embedding

    # combine data
    lowd_conv1 = PCA(n_components=10).fit_transform(conv1)
    lowd_embedding = PCA(n_components=20).fit_transform(embedding)
    
    return lowd_conv1, lowd_embedding


lowd_conv1_full, lowd_embedding_full = load_feature_full(mdir1) # set1

df_full = np.concatenate((lowd_conv1_full, lowd_embedding_full), axis=1)

scaled_df_full = StandardScaler().fit_transform(df_full)


# plot umap
reducer = umap.UMAP(random_state=42, 
                    n_neighbors=20,
                    min_dist=0.0,
                    metric='euclidean',
                    n_components=2)

data_umap_full = reducer.fit_transform(scaled_df_full)

#np.savetxt("R:\\CMPH-Windfarm Field Study\\Duc Phuc Nguyen\\4. Machine Listening\\R script\\UMAP_temporal.csv", data_umap_full, delimiter=",")


