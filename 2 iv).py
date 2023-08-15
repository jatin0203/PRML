#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import matplotlib.pyplot as plt
import random
import math


# In[66]:


X = []
for i in open("./Dataset.csv","r"):
    X.append(list(map(float,i[:-1].split(","))))
X=np.array(X)


# In[67]:


Kmatrix=[]
d=2
for i in range(1000):          
    a =[]
    for j in range(1000):      
        a.append((1+np.inner(X[i],X[j]))**d)      #using the polynomial kernel with degree 2
    Kmatrix.append(a)


# In[68]:


def plot_customised_poly(d,Kmatrix):
    #calculating the eigen values and eigen vectors of the centered K matrix
    eigen_vector_Kp2=np.linalg.eig(Kmatrix)
    #sorting eigen vectors corresponding to highes to lowest eigen values
    eigen_vectors_Kp2=eigen_vector_Kp2[1][:,np.argsort(eigen_vector_Kp2[0])[::-1]]
    #calculating H* based on the top 4 eigen vectors
    Hstar=eigen_vectors_Kp2[:,0:4]
    clust=[-1 for x in range(1000)]
    for i in range(1000):
        clust[i]=np.argmax(Hstar[i])
    color=['Red','Yellow','Green','Blue','pink']
    plt.figure()
    for i in range(1000):
        pnts = np.array(X[i])
        plt.scatter(pnts[0],pnts[1],c=color[clust[i]])
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Customised Clustering d="+str(d))


# In[69]:


plot_customised_poly(d,Kmatrix)

