# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:36:43 2023

@author: madug
"""

import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/madug/Downloads/Zoo.csv")
df
df.head()
df.tail()
df.isna().sum()
df.dtypes
df.shape
df.columns
X=df.iloc[:,1:17]
Y=df.iloc[:,17:]
Y
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_X = MM.fit_transform(X)
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
Train_Acc = []
Test_Acc = []
for i in range(1,200):
    X_train,X_test,Y_train,Y_test = train_test_split(MM_X,Y, test_size=0.3)
    knn = KNeighborsClassifier(n_neighbors=10,p=1) # p=2 --> Eucledian
    knn.fit(X_train,Y_train)
    Y_pred_train = knn.predict(X_train)
    Y_pred_test = knn.predict(X_test)
    ac1 = accuracy_score(Y_train,Y_pred_train)
    ac2 = accuracy_score(Y_test,Y_pred_test)
    Train_Acc.append(ac1)
    Test_Acc.append(ac2)
print("Training accuracy: ", np.mean(ac1.round(2)))
print("Test accuracy: ", np.mean(ac2.round(2)))

#question 2
df=pd.read_csv("C:/Users/madug/Downloads/glass.csv")
df
df.head()
df.tail()
df.isna().sum()
df.dtypes
df.columns
import seaborn as sns
sns.set_style(style="darkgrid")
sns.pairplot(df)
X=df.iloc[:,0:9]
import matplotlib.pyplot as plt
for i in X:
    sns.distplot(X[i])
    plt.figure()
import warnings
warnings.filterwarnings('ignore')

#minmax scaler
Y = df["Type"]
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_X = MM.fit_transform(X)
MM_X

#APPLYING CROSS VALIDATION TECHNIQUES
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
Train_Acc = []
Test_Acc = []
for i in range(1,500):
    X_train,X_test,Y_train,Y_test = train_test_split(MM_X,Y, test_size=0.2)
    knn = KNeighborsClassifier(n_neighbors=3,p=2) # p=2 --> Eucledian
    knn.fit(X_train,Y_train)
    Y_pred_train = knn.predict(X_train)
    Y_pred_test = knn.predict(X_test)
    ac1 = accuracy_score(Y_train,Y_pred_train)
    ac2 = accuracy_score(Y_test,Y_pred_test)
    Train_Acc.append(ac1)
    Test_Acc.append(ac2)
print("Training accuracy: ", np.mean(ac1.round(2)))
print("Test accuracy: ", np.mean(ac2.round(2)))

import numpy as np
from sklearn.model_selection import LeaveOneOut
Loo=LeaveOneOut()
acc1=np.zeros(shape=(214),dtype=float)
acc2=np.zeros(shape=(214),dtype=float)
for train_index, test_index in Loo.split(X):
    X_train, X_test = MM_X[train_index], MM_X[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    m1 = KNeighborsClassifier()
    m1.fit(X_train,Y_train)
    Y_pred_train = m1.predict(X_train)
    Y_pred_test = m1.predict(X_test)
    acc1[train_index] =accuracy_score(Y_train,Y_pred_train)
    acc2[test_index] = accuracy_score(Y_test,Y_pred_test)
print((acc1[train_index]).mean().round(2))
print((acc2[test_index]).mean().round(2))

from sklearn.model_selection import KFold
KF= KFold(n_splits=5)
from sklearn.metrics import accuracy_score
accuracy_train=[]
accuracy_test=[]
for train_index,test_index in KF.split(X):
    X_train, X_test = MM_X[train_index], MM_X[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    m2 = KNeighborsClassifier()
    m2.fit(X_train,Y_train)
    Y_pred_train = m2.predict(X_train)
    Y_pred_test = m2.predict(X_test)
    accuracy_train.append(accuracy_score(Y_train,Y_pred_train))
    accuracy_test.append(accuracy_score(Y_test,Y_pred_test))

print('Training accuracy: ',accuracy_train)
print('Testing accuracy: ',accuracy_test)
print(np.mean(accuracy_train))
print(np.mean(accuracy_test))

#StandardScaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_X = ss.fit_transform(X)
ss_X

#APPLYING CROSS VALIDATION TECHNIQUES under standard scaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
Train_Acc = []
Test_Acc = []
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test = train_test_split(ss_X,Y, test_size=0.2)
    knn1 = KNeighborsClassifier(n_neighbors=2,p=2) # p=2 --> Eucledian
    knn1.fit(X_train,Y_train)
    Y_pred_train = knn1.predict(X_train)
    Y_pred_test = knn1.predict(X_test)
    ac1 = accuracy_score(Y_train,Y_pred_train)
    ac2 = accuracy_score(Y_test,Y_pred_test)
    Train_Acc.append(ac1)
    Test_Acc.append(ac2)
print("Training accuracy: ", np.mean(ac1.round(2)))
print("Test accuracy: ", np.mean(ac2.round(2)))

import numpy as np
from sklearn.model_selection import LeaveOneOut
Loo=LeaveOneOut()
acc1=np.zeros(shape=(214),dtype=float)
acc2=np.zeros(shape=(214),dtype=float)
for train_index, test_index in Loo.split(X):
    X_train, X_test = ss_X[train_index], ss_X[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    m1 = KNeighborsClassifier()
    m1.fit(X_train,Y_train)
    Y_pred_train = m1.predict(X_train)
    Y_pred_test = m1.predict(X_test)
    acc1[train_index] =accuracy_score(Y_train,Y_pred_train)
    acc2[test_index] = accuracy_score(Y_test,Y_pred_test)
print((acc1[train_index]).mean().round(2))
print((acc2[test_index]).mean().round(2))

from sklearn.model_selection import KFold
KF= KFold(n_splits=5)
from sklearn.metrics import accuracy_score
accuracy_train=[]
accuracy_test=[]
for train_index,test_index in KF.split(X):
    X_train, X_test = ss_X[train_index], ss_X[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    m2 = KNeighborsClassifier()
    m2.fit(X_train,Y_train)
    Y_pred_train = m2.predict(X_train)
    Y_pred_test = m2.predict(X_test)
    accuracy_train.append(accuracy_score(Y_train,Y_pred_train))
    accuracy_test.append(accuracy_score(Y_test,Y_pred_test))

print('Training accuracy: ',accuracy_train)
print('Testing accuracy: ',accuracy_test)
print(np.mean(accuracy_train))
print(np.mean(accuracy_test))



















































