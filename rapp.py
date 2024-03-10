import pickle
from typing import Any, List
import time
import numpy as np
import pandas as pd
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

COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
           'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'count', 'srv_count',
           'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
           'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
           'dst_host_srv_rerror_rate']

ATTACK_LABELS = [
    'back.',
    'buffer_overflow.',
    'guess_passwd.',
    'ipsweep.',
    'neptune.',
    'nmap.',
    'portsweep.',
    'rootkit.',
    'satan.',
    'smurf.',
    'spy.',
    'teardrop.',
    'warezclient.',
    'warezmaster.'
]

PROJECT_PATH = "/home/zin/lab/EEProject/app/"
MODEL_PATH = f'{PROJECT_PATH}model'

PROTO_MODEL_NAME = f'{PROJECT_PATH}proto'
SERVICE_MODEL_NAME = f'{PROJECT_PATH}service'
FLAG_MODEL_NAME = f'{PROJECT_PATH}flag'
SC_MODEL_NAME = f'{PROJECT_PATH}sc'
PCA_MODEL_NAME = f'{PROJECT_PATH}pca'
LABEL_MODEL_NAME = f'{PROJECT_PATH}label'

with open(PROTO_MODEL_NAME, 'rb') as f:
    proto_encoder = pickle.load(f)

with open(SERVICE_MODEL_NAME, 'rb') as f:
    service_encoder = pickle.load(f)

with open(FLAG_MODEL_NAME, 'rb') as f:
    flag_encoder = pickle.load(f)

with open(SC_MODEL_NAME, 'rb') as f:
    sc = pickle.load(f)

with open(PCA_MODEL_NAME, 'rb') as f:
    pca = pickle.load(f)

with open(LABEL_MODEL_NAME, 'rb') as f:
    label_encoder = pickle.load(f)

def modelLoder(modelPath: str) -> Any:
    with open(modelPath, 'rb') as f:
        model = pickle.load(f)
    return model

def featureExt(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = df['duration'].astype(float)
    df['protocol_type'] = df['protocol_type'].astype('category')
    df['service'] = df['service'].astype('category')
    df['flag'] = df['flag'].astype('category')
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

    try:
        df['protocol_type'] = proto_encoder.transform(df['protocol_type'].astype(object))
        df['service'] = service_encoder.transform(df['service'].astype(object))
        df['flag'] = flag_encoder.transform(df['flag'].astype(object))
    except ValueError:
        return None
    #print(df)
    X_scaled = sc.transform(df)
    X_pca = pca.transform(X_scaled)
    #print(X_pca)
    return X_pca


def alert(attacks: List[str]) -> None:
    for attack in attacks:
        print(f"Alert! {attack.strip('.')} found!")


def detecter(model: Any, dataFrame: pd.DataFrame):
    X = featureExt(dataFrame)
    if X is None:
        return
    start_time = time.time()
    predictions = model.predict(X)

    results = label_encoder.inverse_transform(predictions.astype(int))
    attacks = [result for result in results if result in ATTACK_LABELS]
    if attacks:
        alert(attacks)
    end_time = time.time()
    #print("Time", end_time-start_time)


def packetScan(model: Any, numOfPacketPerBatch: int) -> None:
    import subprocess
    command = "sudo kdd99extractor"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    data = []
    for line in iter(process.stdout.readline, b''):
        row = line.decode('utf-8').strip().split(',')
        data.append(row)

        if len(data) == numOfPacketPerBatch:
            df = pd.DataFrame(data, columns=COLUMNS)
            detecter(model, df)
            data = []

    process.terminate()



superlearner = modelLoder(MODEL_PATH)
packetScan(superlearner, 32)



"""
X = df.drop(["labels","Subcategories"], axis=1)
y = df["Subcategories"]

ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)


Xtrain_,Xtest_,Ytrain_,Ytest_ = train_test_split(X_pca,y, test_size=0.2, random_state=2, shuffle=True, stratify = y)
"""