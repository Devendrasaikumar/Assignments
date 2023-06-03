# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:00:43 2023

@author: madug
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df =pd.read_csv("C:/Users/madug/Downloads/EastWestAirlines.csv")
df

df.shape

df.head()
df.tail()
df.describe()
df.dtypes

X=df.iloc[:,1:]
for i in X:
    if ((X.dtypes[i])!="O"):
        sns.distplot(X[i])
        plt.figure()
        
        import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

X1= df.iloc[:,1:].values
print(X1)
import matplotlib.pyplot as plt
plt.scatter(X1[:,0],X1[:,1])
plt.show()



import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("airlies")  
dend = shc.dendrogram(shc.linkage(X1, method='complete'))
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='complete')

Y = cluster.fit_predict(X1)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()
plt.figure(figsize=(10, 7))  
plt.scatter(X1[:,0], X1[:,1], c=cluster.labels_, cmap='rainbow') 
 


Y= Y_new[0]
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
ss_count = SS.fit_transform(X1)
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

#K-Means

from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=5)

kmeans = kmeans.fit(X1)

labels = kmeans.predict(X1)

# Total with in centroid sum of squares 
kmeans.inertia_

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

Y=labels
from sklearn.model_selection import train_test_split
acc_train=[]
acc_test=[]
for i in range(1,49):
    X_train,X_test,Y_train,Y_test = train_test_split(ss_count,Y,test_size=(0.3))
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
dbscan = DBSCAN(eps=2.5, min_samples=5)
dbscan.fit(ss_count)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
print(cl['cluster'].value_counts())

clustered = pd.concat([df,cl],axis=1)
noisedata = clustered[clustered['cluster']==-1]
noisedata

finaldata = clustered[clustered['cluster']>=0]
finaldata

Y1= finaldata.iloc[:,12:]
X=finaldata.iloc[:,1:12]
from sklearn.model_selection import train_test_split
acc_train=[]
acc_test=[]
for i in range(1,50):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y1,test_size=(0.3))
    LR2= LogisticRegression()
    LR2.fit(X_train,Y_train)
    Y_train_pred=LR2.predict(X_train)
    Y_test_pred=LR2.predict(X_test)
    acc_train.append(accuracy_score(Y_train,Y_train_pred))
    acc_test.append(accuracy_score(Y_test,Y_test_pred))

print(np.mean(acc_train))
print(np.mean(acc_test))

X1[1]

t = np.array([[28000,0,1,1,1,216,2,0,0,6900,0]])
Y = LR.predict(t)
print(Y)












































