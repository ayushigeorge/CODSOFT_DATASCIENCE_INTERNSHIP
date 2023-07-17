#!/usr/bin/env python
# coding: utf-8

# # CODSOFT INTERNSHIP TASK 2

# # TASK 2 : MOVIE RATING PREDICTION

# In[13]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# # Load the movie review dataset

# In[14]:


df = pd.read_csv('moovy.csv')
df = df.dropna()


# # Split the dataset into features (reviews) and labels (ratings)

# In[15]:


reviews = df['review']
ratings = df['rating']


# # Split the dataset into training and testing sets

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.2, random_state=42)


# # Convert text reviews into numerical feature vectors using CountVectorizer

# In[17]:


vectorizer = CountVectorizer(stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# # Train a Naive Bayes classifier on the training data

# In[18]:


classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)


# # Predict ratings for the test data

# In[19]:


y_pred = classifier.predict(X_test_vectors)


# # Calculate accuracy of the model

# In[20]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[ ]:




