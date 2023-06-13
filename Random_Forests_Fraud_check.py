# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:45:20 2023

@author: madug
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("C:/Users/madug/Downloads/Fraud_check (1).csv")
df
df.head()
df.tail()
df.describe()
df.isna().sum()
df.dtypes
import seaborn as sns
print(sns.countplot(x="Urban", data=df))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Undergrad']=le.fit_transform(df['Undergrad'])
df['Marital.Status']=le.fit_transform(df['Marital.Status'])
df['Urban']=le.fit_transform(df['Urban'])
df
X=df.iloc[:,0:5]
X

import matplotlib.pyplot as plt
import seaborn as sns
for i in X:
    sns.distplot(X[i])
    plt.show()

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

Taxable= []
for i in range(0,600):
    if df["Taxable.Income"][i] <= 30000:
        Taxable.append("1")#risky
    else: 
        Taxable.append("0")# good
Taxable    

df=pd.concat([df,pd.DataFrame({"TN":Taxable})],axis=1)
df

X=df[['Undergrad', 'Marital.Status', 'City.Population','Work.Experience', 'Urban',]]
X

Y=df["TN"]
Y
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_cont  = SS.fit_transform(X)
SS_cont

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
train_acc=[]
test_acc=[]
for i in range(1,600):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_cont,Y, test_size=0.30)

    RFC = RandomForestClassifier(max_samples= 0.6, n_estimators = 100, max_features = 0.6, random_state=24,max_depth=4)
    RFC.fit(X_train,Y_train)
    Y_Pred_train = RFC.predict(X_train)
    Y_Pred_test = RFC.predict(X_test)
    train_acc.append(accuracy_score(Y_train,Y_Pred_train).round(2))
    test_acc.append( accuracy_score(Y_test,Y_Pred_test).round(2))

print('Training accuracy', np.mean(train_acc))
print('Test accuracy', np.mean(test_acc))

from sklearn.ensemble import BaggingClassifier
bag= BaggingClassifier(base_estimator=RFC,max_samples=0.7,n_estimators=100,max_features=0.6,random_state=24)
bag.fit(X_train,Y_train)
Y_Pred_train = bag.predict(X_train)
Y_Pred_test = bag.predict(X_test)
from sklearn.metrics import accuracy_score
print("Training accuracy :",accuracy_score(Y_train,Y_Pred_train).round(2))
print("Test accuracy :",accuracy_score(Y_test,Y_Pred_test).round(2))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
train_acc=[]
test_acc=[]
for i in range(1,600):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_cont,Y, test_size=0.30)

    Rfc = RandomForestClassifier(max_samples= 0.6,criterion='entropy', n_estimators = 100, max_features = 0.6, random_state=24,max_depth=8)
    Rfc.fit(X_train,Y_train)
    Y_Pred_train = Rfc.predict(X_train)
    Y_Pred_test = Rfc.predict(X_test)
    train_acc.append(accuracy_score(Y_train,Y_Pred_train).round(2))
    test_acc.append( accuracy_score(Y_test,Y_Pred_test).round(2))

print('Training accuracy', np.mean(train_acc))
print('Test accuracy', np.mean(test_acc))

from sklearn.ensemble import BaggingClassifier
bag= BaggingClassifier(base_estimator=Rfc,max_samples=0.7,n_estimators=100,max_features=0.6,random_state=24)
bag.fit(X_train,Y_train)
Y_Pred_train = bag.predict(X_train)
Y_Pred_test = bag.predict(X_test)
from sklearn.metrics import accuracy_score
print("Training accuracy :",accuracy_score(Y_train,Y_Pred_train).round(2))
print("Test accuracy :",accuracy_score(Y_test,Y_Pred_test).round(2))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
train_acc=[]
test_acc=[]
for i in range(1,600):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_cont,Y, test_size=0.30)

    RFC = RandomForestClassifier(max_samples= 0.6,criterion='gini', n_estimators = 100, max_features = 0.6, random_state=24,max_depth=8)
    RFC.fit(X_train,Y_train)
    Y_Pred_train = RFC.predict(X_train)
    Y_Pred_test = RFC.predict(X_test)
    train_acc.append(accuracy_score(Y_train,Y_Pred_train).round(2))
    test_acc.append( accuracy_score(Y_test,Y_Pred_test).round(2))

print('Training accuracy', np.mean(train_acc))
print('Test accuracy', np.mean(test_acc))

from sklearn.ensemble import BaggingClassifier
bag= BaggingClassifier(base_estimator=RFC,max_samples=0.7,n_estimators=100,max_features=0.6,random_state=24)
bag.fit(X_train,Y_train)
Y_Pred_train = bag.predict(X_train)
Y_Pred_test = bag.predict(X_test)
from sklearn.metrics import accuracy_score
print("Training accuracy :",accuracy_score(Y_train,Y_Pred_train).round(2))
print("Test accuracy :",accuracy_score(Y_test,Y_Pred_test).round(2))























