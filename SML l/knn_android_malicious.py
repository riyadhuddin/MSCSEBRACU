# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 00:47:28 2019
Detect malware applications (android) through network traffic classification
@author: Riyadh Uddin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
# =============================================================================
# load data file
# URL=('www.kaggle.com/xwolf12/android-malware-analysis/data')
# =============================================================================
dataset = pd.read_csv("android_traffic.csv", sep=";")
# =============================================================================
#dataset.head()
#Data cleaning & processing
#dataset.isna().sum()
#We had to drop empty or NAN columns
#dataset.describe()
#now lets check and clean the duplicates
#dataset[dataset.duplicated()].sum()
# =============================================================================
dataset = dataset.drop(['duracion','avg_local_pkt_rate','avg_remote_pkt_rate','tcp_urg_packet'], axis=1).copy()
dataset=dataset.drop('source_app_packets.1',axis=1).copy()
# =============================================================================
# Feature Scaling
# =============================================================================
scaler = preprocessing.RobustScaler()
scaledData = scaler.fit_transform(dataset.iloc[:,1:11])
scaledData = pd.DataFrame(scaledData, columns=['tcp_packets','dist_port_tcp','external_ips','vulume_bytes','udp_packets','source_app_packets','remote_app_packets',' source_app_bytes','remote_app_bytes','dns_query_times'])
# =============================================================================
# Training and Predictions
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(scaledData.iloc[:,0:10],dataset.type.astype("str"), test_size=0.33, random_state=101)
# =============================================================================
# K Neighbor classification
# Calculating error for K values 
# =============================================================================
error = []
for i in range(3, 15, 3):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    accuracy = accuracy_score(pred_i, y_test)
    print("kneighbors {}".format(i))
    print("Accuracy:",accuracy)
    print("Error:",error)
    print("Confusion Matrix \n",confusion_matrix(y_test, pred_i))
    print("Classification Report: \n",classification_report(y_test, pred_i))

# =============================================================================
# Random Forest Classifier
# =============================================================================
randomForest=RandomForestClassifier(n_estimators=250,
                                    max_depth=50,
                                    bootstrap = True,
                                    random_state=45)
randomForest.fit(X_train,y_train)
pred=randomForest.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print(randomForest)
print("Accuracy:",accuracy)
print("Confusion Matrix \n",confusion_matrix(y_test, pred))
print("Classification Report: \n",classification_report(y_test, pred))
# ============
#xtra coding for roc curve, precision recall will goes here
# =============================================================================
# plt k neighbor error
# =============================================================================
plt.figure(figsize=(12, 6))
plt.plot(range(3,15,3), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
