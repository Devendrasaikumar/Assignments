# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:44:25 2023

@author: madug
"""

import pandas as pd
df = pd.read_csv("C:/Users/madug/Downloads/bank-full.csv",sep=";")
df

df.head()
df.tail()

a=df.dtypes
a

import seaborn as sns
import matplotlib.pyplot as plt
X=df.iloc[:,0:16]
for i in X:
    if ((X.dtypes[i])!="O"):
        sns.distplot(X[i])
        plt.figure()
        
df.columns        

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
for i in range(17):
    if ((df.dtypes[i])=="O"):
        df[df.columns[i]]=le.fit_transform(df[df.columns[i]])      
df

df.shape

df.isna().sum()

df.corr()

X=df.iloc[:,0:16]
Y=df["y"]
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_count = ss.fit_transform(X)
ss_count

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X,Y)
ypred = logreg.predict(X)
from sklearn.metrics import accuracy_score
ac=accuracy_score(Y,ypred)
print("accuracy score",ac)

import numpy as np
a= np.array([[40,2,1,2,0,1002,0,1,2,5,9,261,1,-1,0,3]])
logreg.predict(a)

#Applying Training and testing

np.mean(np.sqrt(acc_train))
np.mean(np.sqrt(acc_test))



















