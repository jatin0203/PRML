#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import random
import math


# In[15]:


Xt = []
for i in open("./Dataset.csv","r"):
    Xt.append(list(map(float,i[:-1].split(","))))
Xt=np.array(Xt)


# In[34]:


error = []
def lloyd(k1):
    k = k1
    clusters = {}
    color=['Red','Yellow','Green','Blue','pink']

    for i in range(k):
      clrmean =  10*(2*np.random.random(2)-1)
      Xp=[]
      cluster = {                                  #creates a structure of clusters which contains mean,points,color of that cluster
           "cluster_mean":clrmean,                #this contains the mean of the cluster
           "points":[],                             #contains poinst assigned to this cluster
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
        plt.title("Clusters with k="+str(k))
#         clusters[i]['points'] = []
    return clusters
total = 0

def cal_clusters(clusters,k):       #assign the points to clusters with mean closest to that particualar point
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

def cluster_mean(clusters,k):                   #calculates the mean of the current allocation of the clusters 
  for i in range(k):
    points = np.array(clusters[i]['points'])
    if points.shape[0]>0:
      new_mean = points.mean(axis=0)
      clusters[i]['cluster_mean'] = new_mean

def termination(clusters,k):                   #terminates the while loop if the error gets convereged after few iteratoins
    global error
    if(len(error) > 2 and error[-1] == error[-2]):       #compares the error with previous iteration
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
def plot_voronoi(k,clusters_val):
    for i in range(k):
        for j in range(1,k):
            xj=clusters_val[j]['cluster_mean'][0]-clusters_val[i]['cluster_mean'][0]
            yj=clusters_val[j]['cluster_mean'][1]-clusters_val[i]['cluster_mean'][1]
            slope=yj/xj
            slope_=(1/(slope))*(-1)
            midpoint=(clusters_val[j]['cluster_mean']+clusters_val[i]['cluster_mean'])/2
            plt.axline(midpoint,slope=slope_)
def euc_dist(p1,p2):
 
    return math.sqrt(((int(p1[0])-int(p2[0]))**2)+((int(p1[1])-int(p2[1]))**2))
def find_cluster(k,clusters_val,pnt):             #find the particular point will be closer to which cluster
    l=0
    for i in range(k):
        new_dist=euc_dist(pnt,clusters_val[i]['cluster_mean'])
        prev_dist=euc_dist(pnt,clusters_val[l]['cluster_mean'])
        if new_dist<prev_dist:
            l=i
    return l;
def voronoi_regions(clusters_val):            #plots the voronoi regions with diff color for diff clusters
    step = 0.5
    i=-10
    while(i<10):
        j = -10
        while(j<10):
            pnt=np.array([i,j])
            l=find_cluster(k,clusters_val,pnt)
            plt.scatter(i,j,c=clusters_val[l]['colors'])
            plt.title("Voronoi Regions")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            j+=step
        i+=step


# In[35]:


k=4
clusters_val=lloyd(k)
plot_voronoi(k,clusters_val)


# In[36]:


voronoi_regions(clusters_val)


# In[37]:


k=3
clusters_val=lloyd(k)
plot_voronoi(k,clusters_val)


# In[38]:


voronoi_regions(clusters_val)


# In[39]:


k=2
clusters_val=lloyd(k)
plot_voronoi(k,clusters_val)


# In[40]:


voronoi_regions(clusters_val)


# In[41]:


k=5
clusters_val=lloyd(k)
plot_voronoi(k,clusters_val)


# In[42]:


voronoi_regions(clusters_val)


# In[ ]:




