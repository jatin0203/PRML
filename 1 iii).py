#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


Xt = []
for i in open("./Dataset.csv","r"):
    Xt.append(list(map(float,i[:-1].split(","))))


# In[3]:


Xt=np.array(Xt);


# In[4]:


#calculating the K matrix in the case of kernel PCA with polynomial kernel
def poly_kernel_pca(k,Xt):
    Kmatrix=[]
    for i in range(1000):          
        a =[]
        for j in range(1000):      
            a.append((1+np.inner(Xt[i],Xt[j]))**k)
        Kmatrix.append(a)
    # centering the data
    onen = (1/1000)*np.ones([Xt.shape[0], Xt.shape[0]])
    mulmatrix=np.eye(Xt.shape[0])-onen
    inter=np.matmul(mulmatrix,Kmatrix)
    Kmatrix_cen=np.matmul(inter,mulmatrix)
    #calculating the eigen values and eigen vectors of the centered K matrix
    eigen_vector_Kp2=np.linalg.eig(Kmatrix_cen)
    #sorting eigen vectors corresponding to highes to lowest eigen values
    eigen_vectors_Kp2=eigen_vector_Kp2[1][:,np.argsort(eigen_vector_Kp2[0])[::-1]]
    #storing top two eigen values
    eigen_values_Kp2=eigen_vector_Kp2[0]
    eigen_values_Kp2=np.sort(eigen_values_Kp2)[::-1]
    #storing eigen vectors corresponding to top eigen values
    alpha_1=eigen_vectors_Kp2[:,0]*(1/(1000*eigen_values_Kp2[0])**0.5)
    alpha_2=eigen_vectors_Kp2[:,1]*(1/(1000*eigen_values_Kp2[1])**0.5)
    #calculating weights acc to two principal components
    weighted_points=[]
    weighted_points.append(Kmatrix_cen@alpha_1);
    weighted_points.append(Kmatrix_cen@alpha_2);
    weighted_points=np.array(weighted_points)
    weighted_points=np.transpose(weighted_points)
    plt.scatter(weighted_points[:,0],weighted_points[:,1])
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    #printing the variance along both principal components
    print(eigen_values_Kp2[0])
    print(eigen_values_Kp2[1])


# In[5]:


poly_kernel_pca(2,Xt)


# In[6]:


poly_kernel_pca(3,Xt)


# In[7]:


def radial_kernel_pca(u,Xt):
    Kmatrix=[]
    for i in range(1000):          
        a =[]
        for j in range(1000):      
            a.append(math.exp((-1*np.inner(Xt[i]-Xt[j],Xt[i]-Xt[j]))/(2*(u**2))))
        Kmatrix.append(a)
    # centering the data
    onen = (1/1000)*np.ones([Xt.shape[0], Xt.shape[0]])
    mulmatrix=np.eye(Xt.shape[0])-onen
    inter=np.matmul(mulmatrix,Kmatrix)
    Kmatrix_cen=np.matmul(inter,mulmatrix)
    #calculating the eigen values and eigen vectors of the centered K matrix
    eigen_vector_Kp2=np.linalg.eig(Kmatrix_cen)
    #sorting eigen vectors corresponding to highes to lowest eigen values
    eigen_vectors_Kp2=eigen_vector_Kp2[1][:,np.argsort(eigen_vector_Kp2[0])[::-1]]
    #storing top two eigen values
    eigen_values_Kp2=eigen_vector_Kp2[0]
    eigen_values_Kp2=np.sort(eigen_values_Kp2)[::-1]
    #storing eigen vectors corresponding to top eigen values
    alpha_1=eigen_vectors_Kp2[:,0]*(1/(1000*eigen_values_Kp2[0])**0.5)
    alpha_2=eigen_vectors_Kp2[:,1]*(1/(1000*eigen_values_Kp2[1])**0.5)
    #calculating weights acc to two principal components
    weighted_points=[]
    weighted_points.append(Kmatrix_cen@alpha_1);
    weighted_points.append(Kmatrix_cen@alpha_2);
    weighted_points=np.array(weighted_points)
    weighted_points=np.transpose(weighted_points)
    return weighted_points,eigen_values_Kp2[0],eigen_values_Kp2[1]


# In[8]:


i=0.1
j=1
plt.figure(figsize=(20,20))
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    plt.subplot(4,3,j)
    plt.tight_layout()
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    wt_point,var_1,var_2=radial_kernel_pca(i,Xt)
    plt.scatter(wt_point[:,0],wt_point[:,1])
    plt.title("Radial Kernel sigma="+str(i))
    i+=i
    j+=1


# In[ ]:




