# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:48:55 2023

@author: madug
"""

import pandas as pd
import numpy as np
df= pd.read_csv("C:/Users/madug/Downloads/crime_data.csv")
df

import warnings
warnings.filterwarnings("ignore")

df.head()
df.tail()

df.info()
df.columns
df.corr()

#since, clustering donot have Target variable in the data set so all the data is considered as X variable only
import seaborn as sns
X=df.iloc[:,:]
for i in X:
    if ((df.dtypes[i])=='O'):
        sns.set_style(style="darkgrid")
        sns.pairplot(df)
        
X= df.iloc[:,1:].values
print(X)
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,3])
plt.show()    

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("CRIME")  
dend = shc.dendrogram(shc.linkage(X, method='complete')) 

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')

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

#Kmeans clustering

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
for i in range(1,49):
    X_train,X_test,Y_train,Y_test = train_test_split(ss_count,Y1,test_size=(0.3))
    LR1= LogisticRegression()
    LR1.fit(X_train,Y_train)
    Y_train_pred=LR1.predict(X_train)
    Y_test_pred=LR1.predict(X_test)
    acc_train.append(accuracy_score(Y_train,Y_train_pred))
    acc_test.append(accuracy_score(Y_test,Y_test_pred))

print(np.mean(acc_train))
print(np.mean(acc_test))

#DBSCAN
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=1, min_samples=4)
dbscan.fit(ss_count)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
print(cl['cluster'].value_counts())

clustered = pd.concat([df,cl],axis=1)
noisedata = clustered[clustered['cluster']==-1]
print(noisedata)
finaldata = clustered[clustered['cluster']>=0]
finaldata

Y1= finaldata.iloc[:,5:]
X=finaldata.iloc[:,1:5]
from sklearn.model_selection import train_test_split
acc_train=[]
acc_test=[]
for i in range(1,49):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y1,test_size=(0.3))
    LR2= LogisticRegression()
    LR2.fit(X_train,Y_train)
    Y_train_pred=LR2.predict(X_train)
    Y_test_pred=LR2.predict(X_test)
    acc_train.append(accuracy_score(Y_train,Y_train_pred))
    acc_test.append(accuracy_score(Y_test,Y_test_pred))

print(np.mean(acc_train))
print(np.mean(acc_test))

import numpy as np
t= np.array([[2.7, 55, 62, 10.9]])
y = LR2.predict(t)
print(y)










































    