#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('diabetes1.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df['Outcome'].value_counts()


# In[ ]:


# 0 = non diabatic
#1 = diabetic


# In[7]:


x = df.drop(columns = 'Outcome',axis=1)
y= df['Outcome']


# In[8]:


x


# In[9]:


y


# In[10]:


scaler = StandardScaler()


# In[12]:


scaler.fit(x)


# In[13]:


standardized_data= scaler.transform(x)


# In[14]:


standardized_data


# In[15]:


x = standardized_data


# In[16]:


x


# In[17]:


y


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2)


# In[20]:


x_train.shape


# In[21]:


x_test.shape


# In[ ]:


#Train the model


# In[23]:


clf = svm.SVC(kernel = 'linear')


# In[24]:


clf.fit(x_train,y_train)


# In[25]:


x_train_prediction = clf.predict(x_train)
accuracy_score(x_train_prediction, y_train)


# In[26]:


#accuracy on test data

x_test_prediction = clf.predict(x_test)
accuracy_score(x_test_prediction,y_test)


# In[27]:


input_sample = (5,166,72,19,175,22.7,0.6,51)


# In[28]:


input_np_array = np.asarray(input_sample)


# In[29]:


input_np_array_reshaped =input_np_array.reshape(1,-1)


# In[30]:


std_data = scaler.transform(input_np_array_reshaped)


# In[31]:


std_data


# In[32]:


prediction =clf.predict(std_data)


# In[33]:


prediction


# In[35]:


if (prediction[0]== 0):
    print("Person is not diabetic")
else:
    print("Person is diabetic")

