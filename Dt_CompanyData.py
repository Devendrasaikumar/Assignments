# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:07:16 2023

@author: madug
"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv("C:/Users/madug/Downloads/Company_Data (1).csv")
df
df.head()
df.tail()
df.dtypes
a=df.columns
a
df.isna()
df.isna().sum()
df["Sales"].max()
df["Sales"].min()
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
for i in a:
    if (df[i].dtypes)!="O":
        sns.distplot(df[i])
        plt.figure()
     
sns.set_style(style="darkgrid")
sns.pairplot(df)     

df.corr()

from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
df['ShelveLoc']=Le.fit_transform(df['ShelveLoc'])
df['Urban']=Le.fit_transform(df['Urban'])
df['US']=Le.fit_transform(df['US'])
df

X=df.iloc[:,1:11]
X
Y = df["Sales"]
Y
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
#DT = DecisionTreeClassifier(criterion='gini')
train_mse=[]
test_mse=[]
DT = DecisionTreeRegressor()
for i in range(1,150):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.30)
    DT.fit(X_train,Y_train)
    Y_Pred_train = DT.predict(X_train)
    Y_Pred_test = DT.predict(X_test)
    train_mse.append(np.sqrt(mse(Y_train,Y_Pred_train).round(2)))
    test_mse.append(np.sqrt(mse(Y_test,Y_Pred_test).round(2)))
print('Training Mse',np.mean(train_mse))
print('Test Mse', np.mean(test_mse)) 

DT.tree_.node_count
Y_Pred_train = DT.predict(X_train)
Y_Pred_test = DT.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
print('Training Mse', np.sqrt(mse(Y_train,Y_Pred_train).round(2)))
print('Test Mse', np.sqrt(mse(Y_test,Y_Pred_test).round(2)))
     
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(DT,filled=True)
graph = graphviz.Source(dot_data)
graph

df
df["Sales"].mean()
df["Sales"][1]

#As per question the sales need to be converted into categorical data so taking out mean and the values equal to mean and above mean are considered to be high and below are considered as low.

Sales_cat = []
for i in range(0,400):
    if df["Sales"][i] >= df["Sales"].mean():
       Sales_cat.append("High")
    else: 
       Sales_cat.append("Low")
df1 = pd.DataFrame({"SA":Sales_cat})
df1
df2=pd.concat([df,df1], axis=1)
df2
X1=df2.iloc[:,1:11]
X1
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
df2["SA"] = Le.fit_transform(df2["SA"])
df2
Y1= df2["SA"]
Y1
from sklearn.tree import DecisionTreeClassifier
DT2 =DecisionTreeClassifier(criterion="gini",max_depth=2)

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_cont  = SS.fit_transform(X1)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
DT1 = DecisionTreeClassifier(criterion='entropy',max_depth=2)
train_acc=[]
test_acc=[]
for i in range(1,150):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_cont,Y1, test_size=0.30)
    DT1.fit(X_train,Y_train)
    Y_Pred_train = DT1.predict(X_train)
    Y_Pred_test = DT1.predict(X_test)
    train_acc.append(accuracy_score(Y_train,Y_Pred_train))
    test_acc.append(accuracy_score(Y_test,Y_Pred_test))
print("train accuracy",np.mean(train_acc))
print("test accuracy",np.mean(test_acc))   

from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(DT1,filled=True)
graph = graphviz.Source(dot_data)
graph









bro idhi googlecolab lo cheyyali.
next?dt froud data kuda 
ssamae ha yes 
 neural net










   