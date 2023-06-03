# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:53:11 2023

@author: madug
"""

import pandas as pd
import numpy as np
df= pd.read_csv("C:/Users/madug/Downloads/wine.csv")
df

pd.options.display.float_format = '{:.2f}'.format
df
df.corr()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style(style='darkgrid')
sns.pairplot(df)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_count = ss.fit_transform(df)
ss_count

from sklearn.decomposition import PCA
pc=PCA()
pca=pc.fit_transform(ss_count)
pca

p = pd.DataFrame(pca)
p

p.columns = ['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3','P1C4']
p

p.iloc[:,0].describe()
p.iloc[:,1].describe()
p.iloc[:,2].describe()
p.iloc[:,3].describe()

l1 = pc.explained_variance_ratio_

l1[0:1]*100
l1[1:2]*100
l1[2:3]*100
l1[3:4]*100

sum(l1[0:3])
sum(l1[5:])


df1 = pd.DataFrame({'var':pc.explained_variance_ratio_,
                  'PC':['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3','P1C4']})


import seaborn as sns
sns.barplot(x='PC',y="var", data=df1, color="c")

df1
l1[0:1]*100
l1[1:2]*100
l1[2:3]*100
sum(l1[5:])
sum(l1[0:3])
x=df1[["PC"]]
X=x.iloc[0:3,:]
X

X= df.iloc[:,1:].values
print(X)
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,3])
plt.show()

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Wine")  
dend = shc.dendrogram(shc.linkage(X, method='complete')) 

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')

Y = cluster.fit_predict(X)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow') 

Y_new[0].value_counts()

Y= Y_new[0]
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
ss_count = SS.fit_transform(X)
ss_count

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
acc_train=[]
acc_test=[]
for i in range(1,102):
    X_train,X_test,Y_train,Y_test = train_test_split(ss_count,Y,test_size=(0.3))
    LR= LogisticRegression()
    LR.fit(X_train,Y_train)
    Y_train_pred=LR.predict(X_train)
    Y_test_pred=LR.predict(X_test)
    acc_train.append(accuracy_score(Y_train,Y_train_pred))
    acc_test.append(accuracy_score(Y_test,Y_test_pred))

print(np.mean(acc_train))
print(np.mean(acc_test))

from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(X)

labels = kmeans.predict(X)

# Total with in centroid sum of squares 
kmeans.inertia_

labels

clust = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)
    
print(clust)

%matplotlib qt
import matplotlib.pyplot as plt
plt.plot(range(1, 11), clust)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

Y1= labels
from sklearn.model_selection import train_test_split
acc_train=[]
acc_test=[]
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(ss_count,Y1,test_size=(0.3))
    LR1= LogisticRegression()
    LR1.fit(X_train,Y_train)
    Y_train_pred=LR1.predict(X_train)
    Y_test_pred=LR1.predict(X_test)
    acc_train.append(accuracy_score(Y_train,Y_train_pred))
    acc_test.append(accuracy_score(Y_test,Y_test_pred))

print(np.mean(acc_train))
print(np.mean(acc_test))











































































































