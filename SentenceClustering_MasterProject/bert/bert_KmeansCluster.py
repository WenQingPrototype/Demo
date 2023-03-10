# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:26:03 2022

@author: 柴文清
"""
import collections
import pandas as pd
#from bert.py import test_data
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
#from Code.Evaluation_Function import *
#import Evaluation_Function.py
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics
import sys
#D:\graduation\code\1\sentence-clustering-master\sentence-clustering-master\Code\Evaluation_Function.py')
import Evaluation_Function
from Evaluation_Function import *
# D:\graduation\code\1\sentence-clustering-master\sentence-clustering-master\Code\Evaluation_Function.py
#feature = np.loadtxt("text_vectors.txt")

#feature = open("../model/bert_vector.txt","rt")
feature = pd.read_csv('../model/bert_array.csv', encoding='utf-8', sep=',')
#text = feature.read()
#m = text.replace(',','')
#k = m.split("\n")
#k = k[:200]
clf = KMeans(n_clusters=4)
s = clf.fit(feature)
score = silhouette_score(feature,s.labels_)
print("silhouette_score",score)


        #print(kmeans.n_iter_)
       # clusters = collections.defaultdict(list)
        #print(kmeans.n_clusters)
        #print(kmeans.cluster_centers_)

        #for i, label in enumerate(kmeans.labels_):
                #print(i,label)
         #       clusters[label].append(i)
        #return dict(clusters),tfidf_matrix 
# 创建遍历，找到最合适的k值
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(feature).labels_
    score = metrics.silhouette_score(feature,labels)
    scores.append(score)
# 通过画图找出最合适的K值
plt.plot(list(range(2,20)),scores)
plt.xlabel('Number of Clusters Initialized')
plt.ylabel('Sihouette Score')
plt.show()
#kn_pre = clf.predict(feature)
 

#birch_pre = Birch(branching_factor=10, n_clusters = 9, threshold=0.5,compute_labels=True).fit_predict(feature)

clusters = collections.defaultdict(list)
print(s.n_clusters)
print(s.cluster_centers_)
def print_clusters(test):
    for i,label in enumerate(test.labels_):
        clusters[label].append(i)
    return dict(clusters)
print_clusters(s)
nclusters=4
#for cluster in range(nclusters):
    #print ("cluster ",cluster,":")
    #for i in enumerate(clusters[cluster]):
        #print ("\tindex of sentence",i)
mylist = []        
for cluster in range(nclusters):
    for i in  clusters[cluster]:
        print([cluster,i])
        mylist.append([cluster,i])
        mylist = sorted(mylist,key=(lambda x:x[1]))
labels = [0]*200
for i in mylist:
    for j in range(200):
        if j == i[1]:
            labels[j]=i[0]
print(labels)
result = [[0]*4 for i in range(4)]

for i in range(200):
    if i < 50:
        for j in range(4):
            if j == labels[i]:
                result[0][j]+=1
    elif i >= 50 and i < 100 :
        for j in range(4):
            if j == labels[i]:
                result[1][j]+=1
    elif i >= 100 and i < 150 :
        for j in range(4):
            if j == labels[i]:
                result[2][j]+=1
    elif i >= 150  :
        for j in range(4):
            if j == labels[i]:
                result[3][j]+=1
print(result) 
print("purity",purity(result))
import numpy as np
C = np.array(result)
print(entropy_total(C))
C_T = C.transpose()
print(entropy_total(C_T))
bert_Kmeans_True = [3]*50 + [1]*50 + [2]*50 + [0]*50 
print(score)
bert_Kmeans_Res = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


bert_Kmeans_True = [3]*50 + [1]*50 + [2]*50 + [0]*50 

f1_micro = f1_score(bert_Kmeans_True,bert_Kmeans_Res,average='micro')
f1_macro = f1_score(bert_Kmeans_True,bert_Kmeans_Res,average='macro')
 
 
print('f1_micro: {0}'.format(f1_micro))
print('f1_macro: {0}'.format(f1_macro))
