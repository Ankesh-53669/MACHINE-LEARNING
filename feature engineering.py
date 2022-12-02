#!/usr/bin/env python
# coding: utf-8

# In[162]:


#importing libraries
import seaborn as sns 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[163]:


#reading dataset
datas=pd.read_csv("Admission_Predict.csv")
df=pd.DataFrame(datas)


# In[164]:


df.head()


# In[165]:


df.shape


# In[166]:


df.columns


# In[167]:


df.info()


# In[168]:


df.describe()


# In[169]:


df.isnull()


# In[170]:


#feature and target
X=df.drop("Chance_of_Admit",axis=1)
x=X.drop("Serial No.",axis=1)
y=df["Chance_of_Admit"]
print(x)
print(y)


# In[171]:


#splitting data for training and testing
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[172]:


#heatmap to check the correlation
plt.figure(figsize=(12,10))
cor=x_train.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.CMRmap_r)
plt.show()


# In[173]:


#creating function for feature selection
def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        
        for j in range (i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                col_name=corr_matrix.columns[i]
                col_corr.add(col_name)
                return col_corr
 
corr_features =correlation(x_train,0.8)
print(len(set(corr_features)))
print(corr_features)

       


# In[174]:


print(x_train)
print(x_test)


# In[177]:


#another way of heatmap
plt.figure()
sns.heatmap(x.corr())
plt.show()


# In[176]:


#lets check the linear relationship between cgpa and chance of admit 
plt.figure()
sns.scatterplot(data=df,x="CGPA" ,y="Chance_of_Admit")
plt.show()




