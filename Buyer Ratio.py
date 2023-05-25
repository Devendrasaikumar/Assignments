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

