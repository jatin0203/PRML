#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import matplotlib.pyplot as plt
import random
import math


# In[111]:


Xt = []
for i in open("./Dataset.csv","r"):
    Xt.append(list(map(float,i[:-1].split(","))))
Xt=np.array(Xt)


# In[112]:


X=[]
X.append(list(Xt[i][0] for i in range(1000)))
Y=[]
Y.append(list(Xt[i][1] for i in range(1000)))
plt.scatter(X,Y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")


# In[113]:


error = []
def lloyd(k1):
    k = k1
    clusters = {}
    color=['Red','Yellow','Green','Blue','pink']

    for i in range(k):
      clrmean =  10*(2*np.random.random(2)-1)
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
        it+=1
        if(not termination(clusters,k) or it>50):
            break;

    plt.figure()
    for i in range(k):
        pnts = np.array(clusters[i]['points'])
        plt.scatter(pnts[:,0],pnts[:,1],c=clusters[i]['colors'])
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        cluster_means = clusters[i]['cluster_mean']
        plt.scatter(cluster_means[0],cluster_means[1],c='black',alpha=0.8)
        plt.text(cluster_means[0],cluster_means[1],"mean")
#         clusters[i]['points'] = []
    return clusters
total = 0

def cal_clusters(clusters,k):
    global total
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
        abserror+=np.min(distance)
        clusters[clstr]['points'].append(point)
    print(abserror)
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


# In[114]:


a=lloyd(4)


# In[115]:


plt.plot(error)
plt.xlabel("Iteration")
plt.ylabel("error")


# In[116]:


a=lloyd(4)


# In[117]:


plt.plot(error)
plt.xlabel("Iteration")
plt.ylabel("error")


# In[118]:


a=lloyd(4)


# In[119]:


plt.plot(error)
plt.xlabel("Iteration")
plt.ylabel("error")


# In[120]:


a=lloyd(4)


# In[121]:


plt.plot(error)
plt.xlabel("Iteration")
plt.ylabel("error")

