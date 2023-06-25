# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 20:01:57 2023

@author: madug
"""

import pandas as pd 
import numpy as np
df =pd.read_csv("C:/Users/madug/Downloads/book.csv")
df

df.head()
df.tail()
df.info()
df.isna().sum()
df.isna()
df.describe()
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

from mlxtend.frequent_patterns import apriori,association_rules

def uni(df):

    for i in range(len(df.columns)):
        print('\n All Unique Value of ' + str(df.columns[i]))
        print(np.sort(df[df.columns[i]].unique()))
        print('Total unique values ' +
                str(len(df[df.columns[i]].unique())))
        uni(df)
        
from matplotlib import pyplot as plt
import seaborn as sns
count = df.sum()
count        

count.sort_values(0, ascending = False, inplace=True)

count = count.to_frame().reset_index()
count = count.rename(columns = {'index': 'items',0: 'count'})
count

sns.barplot(x = 'count', y = 'items', data = count, palette = 'rainbow')
plt.title('Movies Purchase Frequency')


#Aprori Algorithm

from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')
frequent_itemsets=apriori(df,min_support=0.1,use_colnames=True)
frequent_itemsets

frequent_itemsets.sort_values('support',ascending=False)

#with 40% confidence

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.4)
rules

rules.sort_values('lift',ascending=False)


#Sorting the association rules with highest lift ratio for top 20
rules.sort_values('lift',ascending=False)[0:20]
# Lift Ratio > 1 is a good influential rule
rules[rules.lift>1]

plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

#movies

import pandas as pd 
import numpy as np
df =pd.read_csv("C:/Users/madug/Downloads/my_movies.csv")
df

df.drop(columns = ['V1', 'V2', 'V3', 'V4', 'V5'], inplace = True)
df

df.head()
df.tail()
df.isna()
df.info()
df.isna().sum()
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
def uni(df):

    for i in range(len(df.columns)):
        print('\n All Unique Value in ' + str(df.columns[i]))
        print(np.sort(df[df.columns[i]].unique()))
        print('Total no of unique values ' +
                str(len(df[df.columns[i]].unique())))
uni(df)
count = df.sum()
count.sort_values(0, ascending = False, inplace=True)

count = count.to_frame().reset_index()
count = count.rename(columns = {'index': 'items',0: 'count'})
count

from matplotlib import pyplot as plt
import seaborn as sns
sns.barplot(x = 'count', y = 'items', data = count, palette = 'rainbow')
plt.title('Movies Purchase Frequency')

frequent_itemsets=apriori(df,min_support=0.1,use_colnames=True)
frequent_itemsets

rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.8)
rules

rules.sort_values('lift',ascending=False)

rules[rules.lift>1]

plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()