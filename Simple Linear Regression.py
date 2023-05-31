# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:23:39 2023

@author: madug
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/madug/Downloads/delivery_time.csv")
df

df.head()

df.plot(kind="box")

df.tail()

df.isnull().sum()

df.shape



x= df[["Sorting Time"]]
y=df["Delivery Time"]
from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x,y)
Ypred = Lr.predict(x)
from sklearn.metrics import mean_squared_error as mse
Mse= mse(y,Ypred)
print("mean square error",Mse.round(3))
print("root mean square error",np.sqrt(Mse).round(3))
import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color='k')
plt.plot(df["Sorting Time"],Ypred,color="g")
plt.show()

a= int(input())
t = np.array([[a]])
y=(Lr.predict(t)).round(2)

print("output for new x value",y)
#Prediction for new value


#To Add The Predicted Data to the Original Dataset

c = {"Sorting Time":a,"Delivery Time":y[0]}
(pd.DataFrame(c,index=[0]))
df=df.append(c,ignore_index=True)
df

x= df[["Sorting Time"]]
y=df["Delivery Time"]
from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x,y)
Ypred = Lr.predict(x)

#showing new predicted value in the scatterplot

import matplotlib.pyplot as plt
plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"],color='k')
plt.plot(df["Sorting Time"],Ypred,color="g")
plt.show()






#solution for question 2
import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/madug/Downloads/Salary_Data (1).csv")
df

df.tail()
df.head()
df.isnull().sum()
df["YearsExperience"].plot(kind="box")
df["Salary"].plot(kind="box")

x= df[["YearsExperience"]]
y=df["Salary"]
from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x,y)
ypred = Lr.predict(x)
from sklearn.metrics import mean_squared_error as mse
Mse=np.sqrt(mse(y,ypred))
print("root mean square error",Mse.round(3))
print(" mean square error",(mse(y,ypred)).round(3))

import matplotlib.pyplot as plt
plt.scatter(x=df["YearsExperience"],y=df["Salary"],color='k')
plt.plot(df["YearsExperience"],ypred,color="g")
plt.show()

a=int(input())
t = np.array([[a]])
y=(Lr.predict(t)).round(2)
print("output for new x value",y)

#To add the predicted data to the original dataset

c = {"YearsExperience":a,"Salary":y[0]}
(pd.DataFrame(c,index=[0]))
df=df.append(c,ignore_index=True)
df

x= df[["YearsExperience"]]
y=df["Salary"]
from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x,y)
ypred = Lr.predict(x)


#showing new predicted value in the scatterplot

import matplotlib.pyplot as plt
plt.scatter(x=df["YearsExperience"],y=df["Salary"],color='k')
plt.plot(df["YearsExperience"],ypred,color="g")
plt.show()























