#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
df = pd.read_excel("COVID-19.xlsx",usecols=['age','gender','weakness','drowsiness','body temperature'
,'Dry Cough','sour throat','breathing problem','pain in chest','travel history to infected countries','diabetes','heart disease'
,'lung disease','stroke or reduced immunity','symptoms progressed','high blood pressue','kidney disease','change in appetide'
,'Loss of sense of smell','Corona result'])
df.head()


# In[26]:


D = df.values
x = D[:,0:19]
y = D[:,19]


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[29]:


from sklearn.svm import SVC 


# In[30]:


clf = SVC()


# In[31]:


clf.fit(x_train,y_train)


# In[32]:


from sklearn.metrics import confusion_matrix
y_predicted = clf.predict(x_train)
conf_mat = confusion_matrix(y_train,y_predicted)
print (conf_mat)


# In[33]:


#from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_pred = cross_val_predict(clf, x_train, y_train, cv=10)
conf_mat = confusion_matrix(y_train, y_pred)
print (conf_mat)


# In[34]:


clf.predict(x_test)


# In[35]:


from sklearn.model_selection import cross_val_score
for scor in ["recall_weighted", "precision_weighted", "f1_weighted"]:
    print (scor),
    print (" : "),
    print (cross_val_score(SVC(), x_train, y_train, cv=10, scoring=scor).mean())


# In[10]:




