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

columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
           'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'count', 'srv_count',
           'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
           'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
           'dst_host_srv_rerror_rate']


X = df.drop(["labels","Subcategories"], axis=1)
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

MODEL_OBJECT_NAME = '/home/zin/lab/EEProject/app/model'

with open(MODEL_OBJECT_NAME, 'rb') as f:
    superLearner = pickle.load(f)

start_time = time.time()

testX = Xtest_[np.random.choice(Xtest_.shape[0], 4, replace=False)]
result = superLearner.predict(Xtest_)
print(result)

end_time = time.time()
print("Time", end_time-start_time)