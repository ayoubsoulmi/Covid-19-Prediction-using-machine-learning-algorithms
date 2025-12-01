#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
df = pd.read_excel("COVID-19.xlsx")
df.head()


# In[20]:


D = df.values
x = D[:,0:19]
y = D[:,19]


# In[21]:


len(x)
len(y)


# In[22]:


#!pip install imbalanced-learn


# In[23]:


#from imblearn.over_sampling import SMOTE
#sm = SMOTE()
#xsm , ysm = sm.fit_resample(x,y)
#len(xsm)
#len(ysm)


# In[24]:


#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test = train_test_split(xsm,ysm,test_size=0.2)


# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train,y_train)


# In[27]:


from sklearn.metrics import confusion_matrix
y_predicted = clf.predict(x_train)
conf_mat = confusion_matrix(y_train,y_predicted)
print (conf_mat)


# In[28]:


#from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_pred = cross_val_predict(clf, x_train, y_train, cv=10)
conf_mat = confusion_matrix(y_train, y_pred)
print (conf_mat)


# In[29]:


P = clf.predict(x_test)


# In[30]:


#nouveaux données for prediction
out_of_base1 = [0, 1, 101, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
out_of_base2 = [1, 0, 98, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1]
out_of_base3 = [out_of_base2]#, out_of_base2]


# In[31]:


new = clf.predict(out_of_base3)

new


# In[32]:


first = new[0]

first


# In[33]:


P


# In[34]:


len(P)


# In[35]:


clf.score(x_test,y_test)


# In[36]:


from sklearn.model_selection import cross_val_score
for score in ["recall", "precision", "f1"]:
    print (score),
    print (" : "),
    print (cross_val_score(KNeighborsClassifier(), x_train, y_train, cv=10, scoring=score).mean())


# In[19]:


#!pip install anvil-uplink


# In[23]:


import anvil.server

anvil.server.connect("5VWZQRLZHDVR5NVMEQ4BM57L-UULNWN5DJL24Y525")

@anvil.server.callable
def prdiction(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19):
    tab1 = [arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19]
    tab2 = [tab1]
    new = clf.predict(tab2)
    first = new[0]
    if(first == 0):
        return "Negative"
    elif(first == 2):
            return "Positive"  
    else:
        return "Cas non déterminé"
   
    

