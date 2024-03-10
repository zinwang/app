import pickle
import time

import sys
#Importing the library for Machine Learning Model building
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score



import sklearn.model_selection
from mlens.ensemble import SuperLearner


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99
kdd99 = fetch_kddcup99(as_frame=True)['frame']
df = kdd99
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
df['Subcategories'] = df['labels'].str.decode('utf8').apply(lambda r: attacks_types[r[:-1]])
df['Subcategories'] = df['Subcategories'].astype('category')

COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
           'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'count', 'srv_count',
           'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
           'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
           'dst_host_srv_rerror_rate']

CAT_COLUMNS = ['protocol_type', 'service', 'flag', 'land']

from sklearn.preprocessing import LabelEncoder
proto_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()
flag_encoder = LabelEncoder()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
pca = PCA(n_components=0.9)

def featureExt(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = df['duration'].astype(float)
    df['protocol_type'] = df['protocol_type'].str.decode('utf8').astype('category')
    df['service'] = df['service'].str.decode('utf8').astype('category')
    df['flag'] = df['flag'].str.decode('utf8').astype('category')
    df['src_bytes'] = df['src_bytes'].astype(float)
    df['dst_bytes'] = df['dst_bytes'].astype(float)
    df['land'] = df['land'].astype('category')
    df['wrong_fragment'] = df['wrong_fragment'].astype(float)
    df['urgent'] = df['urgent'].astype(float)
    df['count'] = df['count'].astype(float)
    df['srv_count'] = df['srv_count'].astype(float)
    df['serror_rate'] = df['serror_rate'].astype(float)
    df['srv_serror_rate'] = df['srv_serror_rate'].astype(float)
    df['rerror_rate'] = df['rerror_rate'].astype(float)
    df['srv_rerror_rate'] = df['srv_rerror_rate'].astype(float)
    df['same_srv_rate'] = df['same_srv_rate'].astype(float)
    df['diff_srv_rate'] = df['diff_srv_rate'].astype(float)
    df['srv_diff_host_rate'] = df['srv_diff_host_rate'].astype(float)
    df['dst_host_count'] = df['dst_host_count'].astype(float)
    df['dst_host_srv_count'] = df['dst_host_srv_count'].astype(float)
    df['dst_host_same_srv_rate'] = df['dst_host_same_srv_rate'].astype(float)
    df['dst_host_diff_srv_rate'] = df['dst_host_diff_srv_rate'].astype(float)
    df['dst_host_same_src_port_rate'] = df['dst_host_same_src_port_rate'].astype(float)
    df['dst_host_srv_diff_host_rate'] = df['dst_host_srv_diff_host_rate'].astype(float)
    df['dst_host_serror_rate'] = df['dst_host_serror_rate'].astype(float)
    df['dst_host_srv_serror_rate'] = df[ 'dst_host_srv_serror_rate'].astype(float)
    df['dst_host_rerror_rate'] = df['dst_host_rerror_rate'].astype(float)
    df['dst_host_srv_rerror_rate'] = df['dst_host_srv_rerror_rate'].astype(float)

    df['protocol_type'] = proto_encoder.fit_transform(df['protocol_type'])
    df['service'] = service_encoder.fit_transform(df['service'])
    df['flag'] = flag_encoder.fit_transform(df['flag'])


    X_scaled=sc.fit_transform(df)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

# Defining the data also dropping some unnecessary data
X = featureExt(df[COLUMNS])
y = df["Subcategories"]

PROTO_MODEL_NAME = 'proto'
SERVICE_MODEL_NAME = 'service'
FLAG_MODEL_NAME = 'flag'
SC_MODEL_NAME = 'sc'
PCA_MODEL_NAME = 'pca'
with open(PROTO_MODEL_NAME, 'wb') as f:
    pickle.dump(proto_encoder, f)

with open(SERVICE_MODEL_NAME, 'wb') as f:
    pickle.dump(service_encoder, f)

with open(FLAG_MODEL_NAME, 'wb') as f:
    pickle.dump(flag_encoder, f)

with open(SC_MODEL_NAME, 'wb') as f:
    pickle.dump(sc, f)

with open(PCA_MODEL_NAME, 'wb') as f:
    pickle.dump(pca, f)


Xtrain_, Xtest_, Ytrain_, Ytest_ = train_test_split(X, y, test_size=0.2, random_state=2, shuffle=True, stratify = y)

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



