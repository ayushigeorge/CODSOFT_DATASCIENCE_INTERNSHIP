#!/usr/bin/env python
# coding: utf-8

# # CODSOFT DATA SCIENCE INTERNSHIP

# # TASK 1: TITANIC SURVIVAL PREDICTION

# ## Import necessary libraries

# In[10]:


import pandas as pd
import numpy as np
import sys  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ## Load the Titanic dataset

# In[11]:


data = pd.read_csv('tested.csv')


# ## Drop unnecessary columns

# In[12]:


data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)


# ## Handle missing values

# In[13]:


data['Age'].fillna(data['Age'].median(), inplace=True)


# ## Encode categorical variables

# In[14]:


label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])


# ## Replace infinite values with NaN 

# In[20]:


data.replace([np.inf, -np.inf], np.nan, inplace=True)


# ## Drop rows with NaN values

# In[25]:


data.dropna(inplace=True)


# ## Split the data into features and target

# In[26]:


X = data.drop('Survived', axis=1)
y = data['Survived']


# ## Split the data into training and testing sets
# 

# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train a decision tree classifier

# In[28]:


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# ## Make predictions on the test set
# 

# In[29]:


y_pred = clf.predict(X_test)


# ## Evaluate the model

# In[30]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# In[31]:


print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)


# ## Thank you!

# In[ ]:




