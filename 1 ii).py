#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[18]:


Xt = []
for i in open("./Dataset.csv","r"):
    Xt.append(list(map(float,i[:-1].split(","))))


# In[19]:


Xt=np.array(Xt);


# In[20]:


#computing covariance matrix
covar_matrix=(Xt.T@Xt)/1000


# In[22]:


eigen_vector=np.linalg.eig(covar_matrix)
eigen_vector


# In[23]:


eigen_values=np.sort(eigen_vector[0])[::-1]
eigen_values


# In[24]:


sum=np.sum(eigen_values)
print("% of variance covered by first principle component is "+ str((eigen_values[0]/sum)*100)+"%")
print("% of variance covered by second principle component is "+ str((eigen_values[1]/sum)*100)+"%")


# In[25]:


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

