#!/usr/bin/env python
# coding: utf-8

# # CODSOFT DATA SCIENCE INTERNSHIP 

# # TASK 3: IRIS FLOWER DETECTION

# # BY AYUSHI GEORGE

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ## Load the Iris flower dataset

# In[4]:


df = pd.read_csv('IRIS.csv')


# ## Split the dataset into features (measurements) and labels (species)

# In[5]:


X = df.drop('species', axis=1)
y = df['species']


# ## Split the dataset into training and testing sets

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Train a K-Nearest Neighbors classifier on the training data

# In[7]:


classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)


# ## Predict the species for the test data

# In[8]:


y_pred = classifier.predict(X_test)


# ## Calculate accuracy of the model

# In[9]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# # Thank you

# In[ ]:




