# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 17:57:34 2022

@author: 柴文清
"""
import pandas as pd
import nltk
import collections
from sklearn.feature_extraction.text import CountVectorizer
from pylab import mpl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from nltk import word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
import numpy as np
data = open("../Corpus/sentences.txt", "rt")
text = data.read()
data = open("../Corpus/corpus.txt", "rt")
training_data = data.read()
sentences = open("../Corpus/sentences.txt", "rt")
testing_data = sentences.read()
#def word_tokenizer(text):
def cluster(training,testing,nclusters,threshold):
    tfidf_vectorizer = TfidfVectorizer()
    training_matrix = tfidf_vectorizer.fit_transform(training).toarray()
    testing_matrix = tfidf_vectorizer.fit_transform(testing).toarray()
    model = AgglomerativeClustering(distance_threshold=threshold,
                                    compute_distances=True,
                                    n_clusters=nclusters)
    #model = AgglomerativeClustering(n_clusters=3)

    model.fit(training_matrix)
    testing_label = model.fit_predict(testing_matrix)
    print(model.n_clusters_)
    distances = model.distances_
    print(distances.min())
    print(distances.max())
    #clusters = collections.defaultdict(list)
    #for i, label in enumerate(model.labels_):
       # clusters[label].append(i)
    return testing_label,model

        #text = """The Russian military has indicated it will supply the Syrian government with a sophisticated air defence system, after condemning a missile attack launched by the US, Britain and France earlier in April. Col Gen Sergei Rudskoi said in a statement on Wednesday that Russia will supply Syria with new missile defence systems soon. Rudskoi did not specify the type of weapons, but his remarks follow reports in the Russian media that Moscow is considering selling its S-300 surface-to-air missile systems to Syria."""
docs = training_data
    # 切成100句
        #text_list = docs.split("\n")
a = docs.replace("? ?",".")
b = a.replace('; ;','.')
c = b.replace(':','.')
train = nltk.sent_tokenize(c)
test = nltk.sent_tokenize(testing_data)
sents = nltk.sent_tokenize(text)
threshold = None
nclusters = 4
print("First hierararchical clustering")
labels,model = cluster(train,test,nclusters,threshold)