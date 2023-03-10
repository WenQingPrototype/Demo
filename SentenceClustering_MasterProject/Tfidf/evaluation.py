# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:53:54 2022

@author: 柴文清
"""

from sklearn.feature_extraction.text import CountVectorizer
import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import gensim
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
data = open("../Corpus/corpus.txt", "rt")
training_data = data.read()
sentences = open("../Corpus/sentences.txt", "rt")
data = sentences.read()


sents = nltk.sent_tokenize(data)
nclusters= 4

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(sents)
docs = training_data
a = docs.replace("? ?",".")
b = a.replace('; ;','.')
c = b.replace(':','.')
sentence = nltk.sent_tokenize(c)
vectorizer = CountVectorizer()

matrix = tfidf_vectorizer.fit_transform(sents)
weight = tfidf_matrix.toarray()
#for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
    #print(u"-------这里输出第",i,u"类文本的词语tf-idf权重------")
    #for j in range(len(word)):
        #print(word[j],weight[i][j])
        #print(tfidf_matrix.toarray())
        #print("print feature")
        #print(tfidf_vectorizer.get_feature_names())
kmeans = KMeans(n_clusters = 4)
kmeans.fit(tfidf_matrix)
test = kmeans.fit_predict(matrix)
print(test)
print(kmeans.n_iter_)
clusters = collections.defaultdict(list)


fc = [[0]*4 for i in range(4)]

for i in range(200):
    if i < 50:
        for j in range(4):
            if j == test[i]:
                fc[0][j]+=1
    elif i >= 50 and i < 100 :
        for j in range(4):
            if j == test[i]:
                fc[1][j]+=1
    elif i >= 100 and i < 150 :
        for j in range(4):
            if j == test[i]:
                fc[2][j]+=1
    elif i >= 150  :
        for j in range(4):
            if j == test[i]:
                fc[3][j]+=1
print(fc) 
        #print(kmeans.n_clusters)
        #print(kmeans.cluster_centers_)
from sklearn.metrics import fowlkes_mallows_score
score=fowlkes_mallows_score(bert_Kmeans_True,test)

