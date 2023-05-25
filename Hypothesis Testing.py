# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:58:14 2023

@author: madug
"""

import pandas as pd 
import numpy as np
data=pd.read_csv("C:/Users/madug/Downloads/BuyerRatio.csv")
data 

data.drop(['Observed Values'],axis=1,inplace=True)
data

import scipy.stats as stats 
from scipy.stats import chi2_contingency 
p_value=stats.chi2_contingency(data)
p_value

p_value[1]



Costomer+OrderForm

import pandas as pd 
import numpy as np
from scipy import stats
df=pd.read_csv('C:/Users/madug/Downloads/Costomer+OrderForm.csv')
df

df.Phillippines.value_counts()
df.Indonesia.value_counts()
df.Malta.value_counts()
df.India.value_counts()

from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency

# Make a contingency table
df1=np.array([[271,267,269,280],[29,33,31,20]])
df1

# Chi2 contengency independence test
chi2_contingency(df1)


Cutlet

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm

df=pd.read_csv("C:/Users/madug/Downloads/Cutlets.csv")
df

UnitA=pd.Series(df.iloc[:,0])
UnitA

UnitB=pd.Series(df.iloc[:,1])
UnitB

# 2-sample 2-tail ttest:   
p_value=stats.ttest_ind(UnitA,UnitB)
p_value


LabTAT

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm

df=pd.read_csv('C:/Users/madug/Downloads/LabTAT.csv')
df.head()

# Anova ftest statistics: 
p_value=stats.f_oneway(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3])
p_value























