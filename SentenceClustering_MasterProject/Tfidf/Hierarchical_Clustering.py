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
import Evaluation_Function
from Evaluation_Function import *
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score
from sklearn import metrics
data = open("../Corpus/sentences.txt", "rt")
text = data.read()
data = open("../Corpus/corpus.txt", "rt")
training_data = data.read()
sentences = open("../Corpus/sentences.txt", "rt")
testing_data = sentences.read()
#def word_tokenizer(text):
    # tokenizes and stems the text
#    tokens = word_tokenize(text)
#    stemmer = PorterStemmer()
#    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
#    return tokens


def cluster(sentences,nclusters,threshold):
    tfidf_vectorizer = TfidfVectorizer()
    matrix = tfidf_vectorizer.fit_transform(sentences).toarray()
    #testing_matrix = tfidf_vectorizer.fit_transform(testing).toarray()
    model = AgglomerativeClustering(distance_threshold=threshold,
                                    compute_distances=True,
                                    n_clusters=nclusters)
    #model = AgglomerativeClustering(n_clusters=3)

    model = model.fit(matrix)
    #testing_label = model.fit_predict(testing_matrix)
    print(model.n_clusters_)
    score = silhouette_score(matrix,model.labels_)
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
        labels = AgglomerativeClustering(n_clusters=k).fit(matrix).labels_
        score = metrics.silhouette_score(matrix,labels)
        scores.append(score)
# 通过画图找出最合适的K值
    plt.plot(list(range(2,20)),scores)
    plt.xlabel('Number of Clusters Initialized')
    plt.ylabel('Sihouette Score')
    plt.show()
    distances = model.distances_
    print(distances.min())
    print(distances.max())
    #print(model.labels_)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(model.labels_):
        clusters[label].append(i)
    return clusters,model


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    distance = np.arange(model.children_.shape[0])

    linkage_matrix = np.column_stack(
        [model.children_, distance, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.show()



#This works even if distance_threshold is not set.


if __name__ == "__main__":
        #text = """The Russian military has indicated it will supply the Syrian government with a sophisticated air defence system, after condemning a missile attack launched by the US, Britain and France earlier in April. Col Gen Sergei Rudskoi said in a statement on Wednesday that Russia will supply Syria with new missile defence systems soon. Rudskoi did not specify the type of weapons, but his remarks follow reports in the Russian media that Moscow is considering selling its S-300 surface-to-air missile systems to Syria."""

        sents = nltk.sent_tokenize(text)
        threshold = None
        nclusters = 4
        print("First hierararchical clustering")
        clusters,model = cluster(sents,nclusters,threshold)
        plot_dendrogram(model)
        Z = hierarchy.linkage(model.children_, 'ward')
        dn = hierarchy.dendrogram(Z)
        plt.show()

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
print(purity(result))
import numpy as np
C = np.array(result)
print(entropy_total(C))
C_T = C.transpose()
print(entropy_total(C_T))

tfidf_H_Res = [2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 
               1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1,
               1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 1, 1, 
               2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 3, 3, 2, 2, 3, 3, 
               3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
               0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 
               2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 
               2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2]

tfidf_H_True = [1]*50 + [2]*50 + [0]*50 + [3]*50 

f1_micro = f1_score(tfidf_H_True,tfidf_H_Res,average='micro')
f1_macro = f1_score(tfidf_H_True,tfidf_H_Res,average='macro')
print('f1_micro: {0}'.format(f1_micro))
print('f1_macro: {0}'.format(f1_macro))
        #for cluster in range(nclusters):
                #print ("cluster ",cluster,":")
                #for i,sentence in enumerate(clusters[cluster]):
                       # print ("\tsentence ",i,": ",sents[sentence])
        #mylist = []        
        #for cluster in range(nclusters):
            #for i in  clusters[cluster]:
               # print([cluster,i])
               # mylist.append([cluster,i])
               # mylist = sorted(mylist,key=(lambda x:x[1]))
       # q = [0]*200
        #for i in mylist:
            #for j in range(200):
            #    if j == i[1]:
            #        q[j]=i[0]
        #print(q)

        


        #for cluster in range(nclusters):
                #print ("cluster ",cluster,":")
                #for i,sentence in enumerate(clusters[cluster]):
                        #print ("\tsentence ",i,": ",sents[sentence])



#def plot_dendrogram(clusters):
#    plt.figure(figsize=(20,6))
#    dendrogram = hierarchy.dendrogram(clusters, labels=Y, orientation="top",leaf_font_size=9, leaf_rotation=360)
#    plt.ylabel('Euclidean Distance');

#plot_dendrogram(clusters)