#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn import metrics


# In[3]:


df = pd.read_csv('C:\\Users\\admin\\Desktop\\BI\\CApstone\\creditcard.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['Class'].value_counts()


# In[8]:


legal = df[df.Class==0]
fraud = df[df.Class==1]
print(legal.shape)
print(fraud.shape)


# In[9]:


legal.Amount.describe()


# In[10]:


legal_sample= legal.sample(n=492)


# In[11]:


new_df = pd.concat([legal_sample,fraud],axis=0)


# In[12]:


new_df.head()


# In[13]:


new_df.tail()


# In[14]:


new_df['Class'].value_counts()


# In[15]:


x = new_df.drop(columns='Class', axis = 1)
y = new_df['Class']


# In[16]:


print(x)


# In[17]:


print(y)


# In[18]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,stratify=y, random_state=2)


# In[40]:


print(x.shape,x_train.shape,x_test.shape)


# ## Logistic Regression

# In[75]:


model=LogisticRegression()


# In[76]:


model.fit(x_train,y_train)


# In[41]:


x_train_prediction = model.predict(x_train)


# In[42]:


training_data_accuracy = accuracy_score(x_train_prediction,y_train)


# In[43]:


training_data_accuracy


# In[25]:


x_test_prediction = model.predict(x_test)


# In[26]:


testing_data_accuracy= accuracy_score(x_test_prediction,y_test)


# In[27]:


testing_data_accuracy


# In[28]:


cm= confusion_matrix(y_test,x_test_prediction)
print(cm)


# In[32]:


TP = cm[1,1] 
TN = cm[0,0] 
FP = cm[0,1] 
FN = cm[1,0] 


# In[29]:


accuracy_score(y_test,x_test_prediction)


# In[37]:


print("Recall",TP / float(TP+FN))

print("Precision:-", TP / float(TP+FP))


# In[38]:


print(classification_report(y_test,x_test_prediction))


# ## Random Forest

# In[44]:


new_df.shape


# In[46]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=2)


# In[49]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[50]:


rf.fit(x_train,y_train)


# In[62]:


y_test_predicted = rf.predict(x_test)


# In[63]:


testing_data_accuracy_rf= accuracy_score(y_test,y_test_predicted)


# In[64]:


testing_data_accuracy_rf


# In[61]:


cm_rf= confusion_matrix(y_test,x_test_predicted)
print(cm_rf)


# In[65]:


import seaborn as sns

ax = sns.heatmap(cm_rf, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


# In[71]:


TP = cm_rf[1,1] 
TN = cm_rf[0,0] 
FP = cm_rf[0,1] 
FN = cm_rf[1,0] 


# In[72]:


accuracy_score(y_test,y_test_predicted)


# In[73]:


print("Recall",TP / float(TP+FN))

print("Precision:-", TP / float(TP+FP))


# In[74]:


print(classification_report(y_test,y_test_predicted))


# In[ ]:




