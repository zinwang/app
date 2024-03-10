import pickle
import time
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import sys
#Importing the library for Machine Learning Model building
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from mlens.model_selection import Evaluator
from scipy.stats import randint
seed = 2017

import sklearn.model_selection
from mlens.ensemble import SuperLearner


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99
kdd99 = fetch_kddcup99(as_frame=True)['frame']
df = kdd99.stack().str.decode('utf-8').unstack()

attacks_types = {
    'normal': 'normal',
    'back': 'dos',
    'buffer_overflow': 'u2r',
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'ipsweep': 'probe',
    'land': 'dos',
    'loadmodule': 'u2r',
    'multihop': 'r2l',
    'neptune': 'dos',
    'nmap': 'probe',
    'perl': 'u2r',
    'phf': 'r2l',
    'pod': 'dos',
    'portsweep': 'probe',
    'rootkit': 'u2r',
    'satan': 'probe',
    'smurf': 'dos',
    'spy': 'r2l',
    'teardrop': 'dos',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
}

df['Subcategories'] = df['labels'].apply(lambda r:attacks_types[r[:-1]])
df['duration'] = df['duration'].astype('category')
df['protocol_type'] = df['protocol_type'].astype('category')
df['service'] = df['service'].astype('category')
df['flag'] = df['flag'].astype('category')
df['src_bytes'] = df['src_bytes'].astype('category')
df['dst_bytes'] = df['dst_bytes'].astype('category')
df['land'] = df['land'].astype('category')
df['wrong_fragment'] = df['wrong_fragment'].astype('category')
df['urgent'] = df['urgent'].astype('category')
df['hot'] = df['hot'].astype('category')
df['num_failed_logins'] = df['num_failed_logins'].astype('category')
df['logged_in'] = df['logged_in'].astype('category')
df['num_compromised'] = df['num_compromised'].astype('category')
df['root_shell'] = df['root_shell'].astype('category')
df['su_attempted'] = df['su_attempted'].astype('category')
df['num_root'] = df['num_root'].astype('category')
df['num_file_creations'] = df['num_file_creations'].astype('category')
df['num_shells'] = df['num_shells'].astype('category')
df['num_access_files'] = df['num_access_files'].astype('category')
df['num_outbound_cmds'] = df['num_outbound_cmds'].astype('category')
df['is_host_login'] = df['is_host_login'].astype('category')
df['is_guest_login'] = df['is_guest_login'].astype('category')
df['count'] = df['count'].astype('category')
df['srv_count'] = df['srv_count'].astype('category')
df['serror_rate'] = df['serror_rate'].astype('category')
df['srv_serror_rate'] = df['srv_serror_rate'].astype('category')
df['rerror_rate'] = df['rerror_rate'].astype('category')
df['srv_rerror_rate'] = df['srv_rerror_rate'].astype('category')
df['same_srv_rate'] = df['same_srv_rate'].astype('category')
df['diff_srv_rate'] = df['diff_srv_rate'].astype('category')
df['srv_diff_host_rate'] = df['srv_diff_host_rate'].astype('category')
df['dst_host_count'] = df['dst_host_count'].astype('category')
df['dst_host_srv_count'] = df['dst_host_srv_count'].astype('category')
df['dst_host_same_srv_rate'] = df['dst_host_same_srv_rate'].astype('category')
df['dst_host_diff_srv_rate'] = df['dst_host_diff_srv_rate'].astype('category')
df['dst_host_same_src_port_rate'] = df['dst_host_same_src_port_rate'].astype('category')
df['dst_host_srv_diff_host_rate'] = df['dst_host_srv_diff_host_rate'].astype('category')
df['dst_host_serror_rate'] = df['dst_host_serror_rate'].astype('category')
df['dst_host_srv_serror_rate'] = df[ 'dst_host_srv_serror_rate'].astype('category')
df['dst_host_rerror_rate'] = df['dst_host_rerror_rate'].astype('category')
df['dst_host_srv_rerror_rate'] = df['dst_host_srv_rerror_rate'].astype('category')
df['Subcategories'] = df['Subcategories'].astype('category')
label = dict( zip( df["Subcategories"].cat.codes, df["Subcategories"] ) )

df['duration'] = df['duration'].cat.codes
df['protocol_type'] = df['protocol_type'].cat.codes
df['service'] = df['service'].cat.codes
df['flag'] = df['flag'].cat.codes
df['src_bytes'] = df['src_bytes'].cat.codes
df['dst_bytes'] = df['dst_bytes'].cat.codes
df['land'] = df['land'].cat.codes
df['wrong_fragment'] = df['wrong_fragment'].cat.codes
df['urgent'] = df['urgent'].cat.codes
df['hot'] = df['hot'].cat.codes
df['num_failed_logins'] = df['num_failed_logins'].cat.codes
df['logged_in'] = df['logged_in'].cat.codes
df['num_compromised'] = df['num_compromised'].cat.codes
df['root_shell'] = df['root_shell'].cat.codes
df['su_attempted'] = df['su_attempted'].cat.codes
df['num_root'] = df['num_root'].cat.codes
df['num_file_creations'] = df['num_file_creations'].cat.codes
df['num_shells'] = df['num_shells'].cat.codes
df['num_access_files'] = df['num_access_files'].cat.codes
df['num_outbound_cmds'] = df['num_outbound_cmds'].cat.codes
df['is_host_login'] = df['is_host_login'].cat.codes
df['is_guest_login'] = df['is_guest_login'].cat.codes
df['count'] = df['count'].cat.codes
df['srv_count'] = df['srv_count'].cat.codes
df['serror_rate'] = df['serror_rate'].cat.codes
df['srv_serror_rate'] = df['srv_serror_rate'].cat.codes
df['rerror_rate'] = df['rerror_rate'].cat.codes
df['srv_rerror_rate'] = df['srv_rerror_rate'].cat.codes
df['same_srv_rate'] = df['same_srv_rate'].cat.codes
df['diff_srv_rate'] = df['diff_srv_rate'].cat.codes
df['srv_diff_host_rate'] = df['srv_diff_host_rate'].cat.codes
df['dst_host_count'] = df['dst_host_count'].cat.codes
df['dst_host_srv_count'] = df['dst_host_srv_count'].cat.codes
df['dst_host_same_srv_rate'] = df['dst_host_same_srv_rate'].cat.codes
df['dst_host_diff_srv_rate'] = df['dst_host_diff_srv_rate'].cat.codes
df['dst_host_same_src_port_rate'] = df['dst_host_same_src_port_rate'].cat.codes
df['dst_host_srv_diff_host_rate'] = df['dst_host_srv_diff_host_rate'].cat.codes
df['dst_host_serror_rate'] = df['dst_host_serror_rate'].cat.codes
df['dst_host_srv_serror_rate'] = df[ 'dst_host_srv_serror_rate'].cat.codes
df['dst_host_rerror_rate'] = df['dst_host_rerror_rate'].cat.codes
df['dst_host_srv_rerror_rate'] = df['dst_host_srv_rerror_rate'].cat.codes
#df['Subcategories'] = df['Subcategories'].cat.codes

kddDrop = ['hot',
           'num_failed_logins',
           'logged_in',
           'num_compromised',
           'root_shell',
           'su_attempted',
           'num_root',
           'num_file_creations',
           'num_shells',
           'num_access_files',
           'num_outbound_cmds',
           'is_host_login',
           'is_guest_login']

# Defining the data also dropping some unnecessary data
X = df.drop(kddDrop, axis=1)
X = X.drop(["labels", "Subcategories"], axis=1)
y = df["Subcategories"]

ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled=sc.fit_transform(X)
pca = PCA(n_components = 0.9)
X_pca = pca.fit_transform(X_scaled)


Xtrain_,Xtest_,Ytrain_,Ytest_ = train_test_split(X_pca,y, test_size=0.2, random_state=2, shuffle=True, stratify = y)

superlearner = SuperLearner(scorer=accuracy_score, random_state=0, verbose=10, n_jobs = -1, backend="multiprocessing")

# Build the first layer

BaseLayerLR = LogisticRegression(max_iter=1e4,class_weight="balanced",verbose=3,random_state=2,n_jobs=-1)
BaseLayerRF = RandomForestClassifier(n_estimators=10, random_state=0 ,class_weight="balanced",max_features="sqrt",max_depth=5,max_leaf_nodes=10,verbose=3,n_jobs=-1)

IntermediateLayerMLP = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 5), shuffle=True,verbose=3,random_state=2)
IntermediateLayerLD = LinearDiscriminantAnalysis()

MetaLayerLR = LogisticRegression(max_iter=1e4,class_weight="balanced",verbose=3,random_state=2,n_jobs=-1)

superlearner.add([BaseLayerLR,
                  BaseLayerRF
                  ], proba=True)
#SecondLayer, replaced the SVC with decision tree classifier
superlearner.add([IntermediateLayerMLP,
                  IntermediateLayerLD
                  ], proba=True)

# Attach the final meta estimator
superlearner.add_meta(MetaLayerLR)

# Fit ensemble
superlearner.fit(Xtrain_, Ytrain_)

MODEL_OBJECT_NAME = 'model'
with open(MODEL_OBJECT_NAME, 'wb') as f:
    pSuper = pickle.dump(superlearner, f)

size = sys.getsizeof(pSuper)
print("Size", size/1024, "kB")

with open(MODEL_OBJECT_NAME, 'rb') as f:
    super2 = pickle.load(f)

# Predict
preds_superlearner = super2.predict(Xtest_)

#superlearner_acc = accuracy_score(preds_superlearner, Ytest_)
#print("Superlearner prediction score: " + str(superlearner_acc))

