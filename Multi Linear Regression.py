# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:06:28 2023

@author: madug
"""

import pandas as pd
import numpy as np
df= pd.read_csv("C:/Users/madug/Downloads/50_Startups.csv")
df

df.head()

df.tail()

df.shape

df.isna().sum()

X = df.iloc[:,0:4]
Y = df["Profit"]
import matplotlib.pyplot as plt
for i in X:
    plt.scatter(x=df[i],y=Y,color = 'g')
    print(i,plt.show())
    if (i!="State"):
        plt.boxplot(X[i])
        plt.show()
        plt.hist(X[i])
        plt.show()
# since state is categorical value the scatter plot looks different and no box plot is present.
        






#conversion of Categorical values        
        
from sklearn.preprocessing import LabelEncoder
LE =LabelEncoder()
df["State"] = LE.fit_transform(df["State"])
df        
   
df.isna().sum()     
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)        
        
X = df.iloc[:,0:4]
Y = df["Profit"]
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
ss= SS.fit_transform(X)
from sklearn.linear_model import LinearRegression
LR1=LinearRegression()
s=LR1.fit(X,Y)
ypred = LR1.predict(X)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
r2=r2_score(Y,ypred)
Mse= mean_squared_error(Y,ypred)
print('r2',r2)
print("root mean square error",np.sqrt(Mse))        
        
#checking collinearity issues between two variables based on correlation

X=df[["R&D Spend"]]
Y = df["Marketing Spend"]
from sklearn.linear_model import LinearRegression
LR2=LinearRegression()
LR2.fit(X,Y)
ypred = LR2.predict(X)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
r2=r2_score(Y,ypred)
Mse= mean_squared_error(Y,ypred)
print('r2',r2)        
        
        
r2=r2_score(Y,ypred)
vif = 1/(1-r2)
vif        
        
#Testing and training the data

X = df.iloc[:,0:4]
Y = df["Profit"]
from sklearn.linear_model import LinearRegression
LR3=LinearRegression()
from sklearn.model_selection import train_test_split
mse_train=[]
mse_test=[]
for i in range(1,500):
    X_train,X_test,Y_train,Y_test = train_test_split(ss,Y, test_size=0.3)
    k=LR3.fit(X_train,Y_train)
    Y_pred_train=LR3.predict(X_train)
    Y_pred_test = LR3.predict(X_test)
    mse_train.append(mean_squared_error(Y_train,Y_pred_train))
    mse_test.append(mean_squared_error(Y_test,Y_pred_test))


import numpy as np
y = np.mean(np.sqrt(mse_train))
print("mean of all training root mean square errors", y)
a = np.mean(np.sqrt(mse_test))
print("mean of all testing root mean square errors",a)        
        
 df.rename(columns={'R&D Spend':'RD','Administration':'Ad','Marketing Spend':'Ms'},inplace=True)
df       
  
X=df[["RD","Ad","Ms","State"]]
import statsmodels.formula.api as smf
model = smf.ols('Profit~RD+Ad+Ms+State',data=df).fit()
print(model.summary())

import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()

import matplotlib.pyplot as plt
plt.scatter(model.fittedvalues,model.resid)
plt.title('Residual Plot')
plt.xlabel('Fitted values')
plt.ylabel('residual values')
plt.show()
      
        
model_influence = model.get_influence()
(cooks, pvalue) = model_influence.cooks_distance

cooks = pd.DataFrame(cooks)
cooks[0].describe()
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(cooks[0], 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show() 

cooks[0][cooks[0]>0.05]
#(np.argmax(cooks[0]),np.max(cooks[0]))     
        
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()

k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff
        
cooks[0][cooks[0]>leverage_cutoff]
 

       
#user-Interface
        
df['R&D Spend'].describe()        
        
df['Administration'].describe()        
        
df['Marketing Spend'].describe()        



from tkinter import * 
from tkinter import messagebox
from decimal import * 
top=Tk()
top.geometry("600x650")
title=Label(top,text="Profit prediction",font=("Arial",25)) 
title.place(x=100,y=0) 
na=Entry(top)
na.place(x=250,y=50)
RDSpend_label=Label(top,text="R&D Spend :")
RDSpend_label.place(x=150,y=50)
ad=Entry(top)
ad.place(x=250,y=100)
Administration_label=Label(top,text="Administration :")
Administration_label.place(x=150,y=100)

ms=Entry(top)
ms.place(x=250,y=150)
MarketingSpend_label=Label(top,text="Marketing Spend:")
MarketingSpend_label.place(x=150,y=150)

dr=IntVar() 
d1=Radiobutton(top, text="California", variable=dr, value=0) 
d1.place(x=250,y=200)
d2=Radiobutton(top, text="Florida", variable=dr, value=1)
d2.place(x=330,y=200)
d2=Radiobutton(top, text="New York", variable=dr, value=2)
d2.place(x=410,y=200)
drug_label=Label(top,text="Drug :") 
drug_label.place(x=150,y=200)
def action(): 
    if (na.get() == "") or (ad.get() == "") or (ms.get() == ""):
        messagebox.showwarning("warning","All Fields are Required")
    elif(float(na.get()) <= 0.00) or (float(na.get()) >= 165349.20):
        messagebox.showwarning("warning","Enter Valid R&B Value")
    elif(float(ad.get()) <= 51283.14) or (float(ad.get()) >= 182645.56):
        messagebox.showwarning("warning","Enter Valid Administration Value")
    elif(float(ad.get()) <= 0.00) or (float(ad.get()) >= 471784.10):
        messagebox.showwarning("warning","Enter Valid Administration Value")
    
        
btn=Button(top,text="submit",command=action,width=30,height=2) 
btn.place(x=150,y=350)
top.mainloop()




#Corolla data set

import pandas as pd
df=pd.read_csv("C:/Users/madug/Downloads/ToyotaCorolla.csv",encoding="ISO-8859-1")
df = df.loc[:, ["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight","Price"]]
df

df.head()
df.tail()
df[(df["Age_08_04"]==68)].mean()
df.shape
df.columns
df.dtypes
df.corr()

X= df.iloc[:,0:7]
Y=df["Price"]

import matplotlib.pyplot as plt
for i in X:
    plt.scatter(x=X[i],y=Y,color='g')
    print(i,plt.show())
    plt.boxplot(X[i])
    plt.show()
    plt.hist(X[i])
    plt.show()
    print("-----------------------------------------------------")

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

X=df[['Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears','Quarterly_Tax','Weight']]
Y=df["Price"]
from sklearn.preprocessing import StandardScaler
SS =StandardScaler()
SS_count = SS.fit_transform(X)
SS_count


#checking for multicollinearity isses

X=df[['Weight']]
Y = df['Quarterly_Tax']
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
ypred = LR.predict(X)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
r2=r2_score(Y,ypred)
Mse= mean_squared_error(Y,ypred)
print('r2',r2)

r2=r2_score(Y,ypred)
vif = 1/(1-r2)
vif

X=df[['Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears','Quarterly_Tax','Weight']]
Y=df["Price"]
from sklearn.preprocessing import StandardScaler
SS =StandardScaler()
SS_count = SS.fit_transform(X)
SS_count

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_count,Y, test_size=0.3)
mse_train=[]
mse_test=[]
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_count,Y, test_size=0.3)
    LR.fit(X_train,Y_train)
    Y_pred_train=LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    mse_train.append(mean_squared_error(Y_train,Y_pred_train))
    mse_test.append(mean_squared_error(Y_test,Y_pred_test))
print(mse_train)
print(mse_test)
import numpy as np
y = np.mean(np.sqrt(mse_train))
print(y)
a = np.mean(np.sqrt(mse_test))
print(a)

import seaborn as sns
X = df.iloc[:,0:7]

df.isna().sum()
for i in X:
    if ((X.dtypes[i])!="O"):
        X[i] = np.log(X[i])
X

Y=df["Price"]
from sklearn.model_selection import train_test_split
mse_train=[]
mse_test=[]
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3)
    LR.fit(X_train,Y_train)
    Y_pred_train=LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    mse_train.append(mean_squared_error(Y_train,Y_pred_train))
    mse_test.append(mean_squared_error(Y_test,Y_pred_test))
print(mse_train)
print(mse_test)


import numpy as np
y = np.mean(np.sqrt(mse_train))
print("mean of all training root mean square errors", y)
a = np.mean(np.sqrt(mse_test))
print("mean of all testing root mean square errors",a)

import seaborn as sns
for i in X:
    sns.distplot(X[i])
    plt.figure()
import warnings
warnings.filterwarnings('ignore')
        


import statsmodels.formula.api as smf
model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()
print(model.summary())
#R squared values
(model.rsquared.round(3),model.rsquared_adj.round(3))

import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()

import matplotlib.pyplot as plt
plt.scatter(model.fittedvalues,model.resid)
plt.title('Residual Plot')
plt.xlabel('Fitted values')
plt.ylabel('residual values')
plt.show()

model_influence = model.get_influence()
(cooks, pvalue) = model_influence.cooks_distance

cooks = pd.DataFrame(cooks)
cooks[0].describe()
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(cooks[0], 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()

cooks[0][cooks[0]>0.05]
#(np.argmax(cooks[0]),np.max(cooks[0]))

from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()

k = df.shape[1]
n = df.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

a=[cooks[0][cooks[0]>leverage_cutoff]]
a

df.drop([14,16,80,109,110,111,141,191,192,221,523,601,654,960,991,1058],inplace=True)
df.shape
df


















