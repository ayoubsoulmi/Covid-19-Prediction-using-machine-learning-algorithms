#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_excel("COVID-19.xlsx")
df.head()


# In[3]:


x = df[['age','gender','weakness','drowsiness','body temperature' ,'Dry Cough','sour throat','breathing problem','pain in chest','travel history to infected countries','diabetes','heart disease' ,'lung disease','stroke or reduced immunity','symptoms progressed','high blood pressue','kidney disease','change in appetide' ,'Loss of sense of smell']]

y = df ['Corona result']


# In[4]:


x


# In[5]:


y


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


x_train,x_test,y_train,y_ttest = train_test_split(x,y,test_size=0.2)
y_train


# In[8]:


len(x_train)


# In[9]:


len(x_test)


# In[10]:


from sklearn.linear_model import LinearRegression
clf  = LinearRegression()


# In[11]:


clf.fit(x_train,y_train)


# In[13]:


cle = []
for j in range(0,len(y_train)):
    cle.insert(j,0)
i = 0
for val in y_train.keys():
    cle[i] = val
    i = i + 1
    
print(cle)   
len(cle)


# In[58]:


for l in range(0,len(cle)):
    print(y_train.get(cle[l]))


# In[1]:


cle.clear()
    


# In[15]:


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, x_train, y_train, cv=3)
print (y_pred)


# In[15]:


predtab = []
erreur = clf.predict(x_train)
A = clf.predict(x_train)
for j in range(0,len(y_train)):
    predtab.insert(j,0)
print (erreur)
for i in range(0,len(y_train)):
    A[i] = erreur[i] + y_train.get(cle[i])
    if A[i] < 0.5:
        predtab[i] = 0
    elif A[i] < 1.5:
        predtab[i] = 1
    else:
        predtab[i] = 2
        
from sklearn.metrics import confusion_matrix
#y_predicted = clf.predict(x_train)
conf_mat = confusion_matrix(y_train,predtab)
print (conf_mat)


# In[68]:


predtab.clear()


# In[74]:


clf.predict(x_test)


# In[31]:


y_ttest
len(y_ttest)


# In[76]:


clf.score(x_test,y_ttest)


# In[83]:


#from sklearn.model_selection import cross_validate
#cross_validate.


# In[96]:


from sklearn.model_selection import cross_val_score
for score in ["recall", "precision", "f1"]:
    print (score),
    print (" : "),
    print (cross_val_score(LinearRegression(), x_train, y_train, cv=5, scoring=score).mean())

