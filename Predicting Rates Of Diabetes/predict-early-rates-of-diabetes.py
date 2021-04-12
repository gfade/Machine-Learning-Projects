#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:



import numpy as np 
import pandas as pd 
import torch as t
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# In[5]:


model = Sequential()
model.add(Dense(12, input_dim=16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[6]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=10)
accuracy = model.evaluate(X_test, y_test, verbose=0)
print(accuracy)


# In[ ]:




