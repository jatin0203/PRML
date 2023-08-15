#!/usr/bin/env python
# coding: utf-8

# In[374]:


import numpy as np
import pandas as pd
import re
import glob
from tqdm import tqdm
import io


# In[375]:


mails = pd.read_csv('./smsspamcollection/SMSSpamCollection', sep='\t',names=["label", "email"])
stopwords= [i[:-1] for i in open('./stopwords.txt','r')]


# In[376]:


url = "test/*"
files = [file for file in glob.glob(url)]

test = []

for file_name in tqdm(files):
        with io.open(file_name, 'r') as f:
            mail = f.read()
            test.append(mail)

test = np.array(test)

df = pd.DataFrame(test)
df["text"] = df[0]
df = df.drop([0],axis=1)


# In[377]:


test


# In[378]:


mails


# In[379]:


corpus = []
for i in range(0, len(mails)):
    words = re.sub('[^a-zA-Z]',' ',mails['email'][i])
    words = words.lower()
    words = words.split()
    words = [word for word in words if not word in stopwords]
    words = ' '.join(words)
    corpus.append(words)


# In[380]:


test_corpus = []
for i in range(0, len(test)):
    testwords = re.sub('[^a-zA-Z]',' ',test[i])
    testwords = testwords.lower()
    testwords = testwords.split()
    testwords = [word for word in testwords if not word in stopwords]
    test_corpus.append(testwords)


# In[381]:


corpus


# In[382]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(corpus).toarray()


# In[383]:


for i in X:
    for j in i:
        if(j>1):
            j=1;


# In[384]:


features=cv.get_feature_names()
print(features)


# In[385]:


X_test = [[0]*3000]*len(test_corpus)
for i in range(0, len(test_corpus)):
    for word in test_corpus[i]:
        for j in range(3000):
            if(features[j]==word):
                X_test[i][j]=1
print(X_test)


# In[386]:


X.shape


# In[387]:


max(X_test[4])


# In[388]:


y=pd.get_dummies(mails['label'])     #create two columns spam and ham only carry spam label which tells whether spam or ham
y=y.iloc[:,1]
y.shape


# In[389]:


from sklearn.model_selection import train_test_split        #split the data into 80-20 ratio for train and test
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[390]:


def posterior(y,X_test,Pj1,Pj0,p,i):
    if(y==1):
        Pj=Pj1
        py=p     #assign probability of y being 1
    else:
        Pj=Pj0
        py=1-p   #assign probability of y being 0 
    d=X_test.shape[1]
    P=1
    for j in range(d):    #for all the features calculate the posterior 
        P=P*pow(Pj[j],X_test[i][j])*pow(1-Pj[j],1-X_test[i][j])
    P=P*py
    return P
def naiveBayes(X_test,y_train,X_train):
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)   
    d=X_train.shape[1]
    n=X_train.shape[0]
    county=0
    for i in range(n):
        if(y_train[i]==1):
            county=county+1     #count number of spam emails and assign p that value
    p=county/n
    Pj1=[]
    Pj0=[]
    for j in range(d):     #for every word check how many times it gets appeared if it was a spam or a ham email
        count1=0
        count0=0
        for i in range(n):
            if(y_train[i]==1 and X_train[i][j]==1):
                count1=count1+1
            elif(y_train[i]==0 and X_train[i][j]==1):
                count0=count0+1
        Pj1.append(count1/county)
        Pj0.append((count0/(n-county)))
    y_pred=[]          #predict the values of y for all mails in the X_test
    for i in range(X_test.shape[0]):
        if(posterior(1,X_test,Pj1,Pj0,p,i)>=posterior(0,X_test,Pj1,Pj0,p,i)):  #compare the posterior using prior and assign y
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


# In[391]:


y_pred=naiveBayes(x_test,y_train,X_train)
spam_pred=naiveBayes(X_test,y,X)


# In[392]:


def score(y_pred,y_test):
    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    count=0
    for i in range(len(y_test)):
        if (y_pred[i] == y_test[i]):
            count=count+1
    print(count)
    return (count/len(y_test))*100
y_test.shape


# In[393]:


score(y_pred,y_test)


# In[394]:


spam_pred


# In[ ]:




