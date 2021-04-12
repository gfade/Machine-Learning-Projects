#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



import numpy as np 
import graphviz 
import pandas as pd 
import torch as t
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os


# In[2]:


data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv')
data.isnull().sum()
data.head()


# In[3]:


le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)
data.head()


# In[4]:


data.dtypes
X = data.iloc[:,:-1]
y = data[data.columns[-1]]


# In[5]:


tr = tree.DecisionTreeClassifier()
tr= tr.fit(X,y)


# In[6]:


dot_data = tree.export_graphviz(tr, out_file=None, 
                      feature_names=data.columns[:-1],  
                      class_names= data.columns[-1],  
                      filled=True, rounded=True,  
                      special_characters=False)
graph = graphviz.Source(dot_data) 
graph

