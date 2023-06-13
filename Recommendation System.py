# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:30:14 2023

@author: madug
"""

import pandas as pd
import numpy as np

df =pd.read_csv("C:/Users/madug/Downloads/book (1).csv",encoding='latin1')
df

df.head()
df.tail()
df.shape
df.isnull().sum()
df1 = df.rename(columns={'User.ID':'UserID','Book.Title':'BookTitle','Book.Rating':'BookRating'})
df1

df1.sort_values('UserID')
df.iloc[:,1:]
len(df1.UserID.unique())
len(df1.BookTitle.unique())
user_df = df1.pivot_table(index='UserID',columns='BookTitle',values='BookRating')
user_df
user_df.fillna(0,inplace=True)
user_df

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
user_sim=1-pairwise_distances(user_df.values,metric='cosine')
user_sim

# store the reults in DataFrame
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index   = df1.UserID.unique()
user_sim_df.columns = df1.UserID.unique()
user_sim_df

np.fill_diagonal(user_sim, 0)
user_sim_df.max()
# most similar users
#Most Similar Users
user_sim_df.max().sort_values(ascending=False).head(50)
user_sim_df.idxmax(axis=1)[86]
df[(df1['UserID']==3048) | (df1['UserID']==278806)]


























