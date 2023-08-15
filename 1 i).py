#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[47]:


Xt = []
for i in open("./Dataset.csv","r"):
    Xt.append(list(map(float,i[:-1].split(","))))


# In[48]:


Xt=np.array(Xt);
#After centering the data to its mean
Xt=Xt-np.mean(Xt,axis=0);


# In[50]:


Xt


# In[59]:


#computing covariance matrix
covar_matrix=(Xt.T@Xt)/1000


# In[60]:


covar_matrix


# In[61]:


eigen_vector=np.linalg.eig(covar_matrix)


# In[62]:


eigen_vector


# In[63]:


eigen_values=np.sort(eigen_vector[0])[::-1]
eigen_values


# In[64]:


sum=np.sum(eigen_values)
print("% of variance covered by first principle component is "+ str((eigen_values[0]/sum)*100)+"%")
print("% of variance covered by second principle component is "+ str((eigen_values[1]/sum)*100)+"%")


# In[65]:


principal_components=eigen_vector[1][:,np.argsort(eigen_vector[0])[::-1]]
principal_components


# In[66]:


X=[]
X.append(list(Xt[i][0] for i in range(1000)))
Y=[]
Y.append(list(Xt[i][1] for i in range(1000)))
plt.scatter(X,Y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.plot(x,y)
xy = (0,0)
xy1 = (-0.323516,-0.9462227)
xy2 = (-0.9462227,0.323516)
plt.axline(xy,xy2,color='r')
plt.axline(xy,xy1)
plt.legend(["Points","PC1","PC2"])

