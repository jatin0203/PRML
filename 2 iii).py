#!/usr/bin/env python
# coding: utf-8

# In[244]:


import numpy as np
import matplotlib.pyplot as plt
import random
import math


# In[245]:


X = []
for i in open("./Dataset.csv","r"):
    X.append(list(map(float,i[:-1].split(","))))
X=np.array(X)


# In[246]:


#2 iii)
Kmatrix=[]
d=2
for i in range(1000):          
    a =[]
    for j in range(1000):      
        a.append((1+np.inner(X[i],X[j]))**d)
    Kmatrix.append(a)
# centering the data
onen = (1/1000)*np.ones([X.shape[0], X.shape[0]])
mulmatrix=np.eye(X.shape[0])-onen
inter=np.matmul(mulmatrix,Kmatrix)
Kmatrix_cen=np.matmul(inter,mulmatrix)
#calculating the eigen values and eigen vectors of the centered K matrix
eigen_vector_Kp2=np.linalg.eig(Kmatrix_cen)
#sorting eigen vectors corresponding to highes to lowest eigen values
eigen_vectors_Kp2=eigen_vector_Kp2[1][:,np.argsort(eigen_vector_Kp2[0])[::-1]]
Hstar=eigen_vectors_Kp2[:,0:4]
Hstar.shape


# In[247]:


for i in range(1000):
    Hstar[i]=Hstar[i]/np.linalg.norm(Hstar[i])


# In[248]:


Hstar[0]


# In[249]:


Xt=Hstar
print(Xt[0].sum())


# In[250]:


error = []
clust = []
def lloyd(k1):
    global clust
    clust = [-1 for x in range(1000)]
    k = k1
    clusters = {}
    color=['Red','Yellow','Green','Blue','pink']

    for i in range(k):
      clrmean =  (2*np.random.random(4)-1)
      Xp=[]
      cluster = {
           "cluster_mean":clrmean,
           "points":[],
           "colors":color[i]
      }
      clusters[i] = cluster
    global error
    error=[]
    #error.clear()
    it=0
    while(True):
        cal_clusters(clusters,k)
        cluster_mean(clusters,k)
#         print(clust)
        it+=1
        if(not termination(clusters,k) or it>50):
            break;

#     plt.figure()
#     for i in range(k):
#         pnts = np.array(clusters[i]['points'])
#         plt.scatter(pnts[:,0],pnts[:,1],c=clusters[i]['colors'])
#         plt.xlabel("X-axis")
#         plt.ylabel("Y-axis")
#         cluster_means = clusters[i]['cluster_mean']
#         plt.scatter(cluster_means[0],cluster_means[1],c='black',alpha=0.8)
#         plt.text(cluster_means[0],cluster_means[1],"mean")
        clusters[i]['points'] = []
    return clusters
total = 0

def cal_clusters(clusters,k):
    global total
    global clust
    global error
    
    abserror=0;
    for j in range(k):
        clusters[j]['points']=[]
    for i in range(Xt.shape[0]):
        distance = []
        point = Xt[i]
        for j in range(k):
          dist = math.dist(point,clusters[j]['cluster_mean'])
          distance.append(dist)
        clstr = np.argmin(distance)
        clust[i]=clstr
        abserror+=np.min(distance)
        clusters[clstr]['points'].append(point)
    error.append(abserror)
    total += 1

def cluster_mean(clusters,k):
  for i in range(k):
    points = np.array(clusters[i]['points'])
    if points.shape[0]>0:
      new_mean = points.mean(axis=0)
      clusters[i]['cluster_mean'] = new_mean

def termination(clusters,k):
    global error
    if(len(error) > 2 and error[-1] == error[-2]):
        return False
    temp=clusters.copy()
    for j in range(k):
        clusters[j]['points']=[]
    for i in range(Xt.shape[0]):
        distance = []
        point = Xt[i]
        for j in range(k):
            dist = math.dist(point,clusters[j]['cluster_mean'])
            distance.append(dist)
        clstr = np.argmin(distance)
        clust[i]=clstr
        clusters[clstr]['points'].append(point)
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0]>0:
          new_mean = points.mean(axis=0)
          clusters[i]['cluster_mean'] = new_mean
    flag=True
    for l in range(k):
        if(clusters[l]['cluster_mean'] is not temp[l]['cluster_mean']):
            flag=False
        if(flag is False):
            return False
    return True


# In[251]:


cluster_val=lloyd(4)


# In[252]:


plt.figure()
for i in range(1000):
    pnts = np.array(X[i])
    plt.scatter(pnts[0],pnts[1],c=cluster_val[clust[i]]['colors'])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Spectral Clustering with polynomial kernel d="+str(d))

