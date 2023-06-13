# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:15:17 2023

@author: madug
"""

import pandas as pd
df1_train = pd.read_csv("C:/Users/madug/Downloads/SalaryData_Train.csv")
df1_train

df1_test = pd.read_csv("C:/Users/madug/Downloads/SalaryData_Test.csv")
df1_test

frames = [df1_train, df1_test]
df = pd.concat(frames, ignore_index=True)
df
df.dtypes
df.columns
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

import seaborn as sns
import matplotlib.pyplot as plt
X=df.iloc[:,0:14]
for i in X:
    if ((X.dtypes[i])!="O"):
        sns.distplot(X[i])
        plt.figure()
        
from sklearn.preprocessing import LabelEncoder
for i in range(0,14):
    if ((df1_train.dtypes[i])=="O"):
        df1_train[df1_train.columns[i]]=LabelEncoder().fit_transform(df1_train[df1_train.columns[i]])      
df1_train 

for i in range(0,14):
    if ((df1_test.dtypes[i])=="O"):
        df1_test[df1_test.columns[i]]=LabelEncoder().fit_transform(df1_test[df1_test.columns[i]])      
df1_test

X_train = df1_train.iloc[:,0:13]
Y_train =df1_train.iloc[:,13:]
X_test =df1_test.iloc[:,0:13]
Y_test =df1_test.iloc[:,13:]

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

Y_pred_train = MNB.predict(X_train)
Y_pred_test = MNB.predict(X_test)

from sklearn.metrics import accuracy_score

acc1 = accuracy_score(Y_train,Y_pred_train).round(2)
print("naive bayes model Training score:" , acc1)
acc2 = accuracy_score(Y_test,Y_pred_test).round(2)
print("naive bayes model Test score:" , acc2)



















       