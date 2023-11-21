#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv("datasets/Churn_Modelling.csv")


# In[3]:


df.columns


# In[4]:


df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


# In[5]:


df.head(10)


# In[6]:


label_encoder = LabelEncoder()
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])


# In[7]:


df.head(10)


# In[9]:


X = df.drop('Exited', axis=1)
y = df['Exited']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', random_state=42)
model.fit(X_train, y_train)


# In[13]:


y_pred = model.predict(X_test)


# In[15]:


print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")


# In[ ]:




