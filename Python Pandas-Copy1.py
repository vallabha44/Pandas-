#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
a = {'student-name':['bobby','Oliver','Harry','George','Jack','Leo'],
    '1st-sem':[45,56,47,67,35,53]}
df = pd.DataFrame(a)
print(df)


# In[3]:


df.shape


# In[4]:


df.index


# In[5]:


df.size


# In[6]:


df.columns


# In[7]:


df.values


# In[8]:


df.axes


# In[9]:


df.ndim


# In[10]:


df['2nd-sem']=[34,36,45,67,63,55]
df


# In[11]:


df.columns=['stdnt-name','first-sem','second-sem']
df


# # Pandas Methods

# In[12]:


df.count()


# In[13]:


df.sum()


# In[14]:


df.max()


# In[15]:


df.describe()


# In[16]:


df.head()


# In[17]:


df.head(3)


# In[18]:


df.tail()


# In[19]:


df.tail(4)


# In[20]:


df.mean()


# In[21]:


df.std()


# In[22]:


df.sample()


# In[23]:


df.sample(n=2)


# In[24]:


df.sample(n=5)


# In[25]:


df.median()


# In[26]:


df.sort_values(by='first-sem',ascending=False)


# In[27]:


df.sort_values(by='first-sem')


# In[28]:


df.loc[0]


# In[29]:


df.loc[5]


# In[30]:


df.iloc[:,-1]


# In[31]:


df.iloc[:,:]


# In[32]:


import numpy as np
df['third-sem']=[34,56,np.nan,35,55,47]
df


# In[33]:


df.isna()


# In[34]:


df.notnull()


# In[35]:


df.fillna(45)


# In[36]:


df.drop(3)


# In[37]:


df.dropna()


# In[38]:


df.dropna(axis=1,how='any')


# In[39]:


df.dropna(axis=0,how='any')


# In[40]:


df.dropna(axis=0,how='all')


# In[114]:


x = {'stdnt-name':['bobby','Oliver','Harry','George','Jack','Leo'],
    'first-sem':[45,56,47,67,35,53],
    'second-sem':[37,46,70,51,49,54],
    'third-sem':[51,43,61,45,37,62]}
y = {'stdnt-name':['charlie','butcher','hughie','MM','Kimiko','Homelander'],
    'first-sem':[35,46,57,59,68,39],
    'second-sem':[37,46,70,51,49,54],
    'third-sem':[51,43,61,45,37,62]}
df = pd.DataFrame(x,index=[0,1,2,3,4,5])
de = pd.DataFrame(y,index=[6,7,8,9,10,11])
c = pd.concat([df,de])
c


# In[115]:


result = df.append(de)
result


# In[116]:


result.groupby(['first-sem','second-sem']).groups


# In[117]:


result.insert(4,'fourth-sem',[51,43,61,45,37,62,51,43,61,45,37,62])
result


# In[118]:


result.insert(5,'fifth-sem',[51,43,61,45,37,62,51,43,61,45,37,62])
result


# In[122]:


result.rename(columns={'fifth-sem':'5th-sem'})


# In[123]:


result.assign(sixthsem=[45,56,47,67,35,53,45,56,47,67,35,53],
             seventhsem=[45,56,47,67,35,53,45,56,47,67,35,53],
             eighthsem=[45,56,47,67,35,53,45,56,47,67,35,53])


# In[121]:


result.groupby(['third-sem']).agg([min,max])


# In[ ]:




