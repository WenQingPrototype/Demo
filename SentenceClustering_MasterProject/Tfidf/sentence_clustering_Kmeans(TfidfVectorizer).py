import collections
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import gensim
#sys.path.append("D:\graduation\code\1\sentence-clustering-master\sentence-clustering-master\Code")
import sys
#D:\graduation\code\1\sentence-clustering-master\sentence-clustering-master\Code\Evaluation_Function.py')
import Evaluation_Function
from Evaluation_Function import *
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score
from sklearn import metrics
#from Code.Evaluation_Function import *
data = open("../Corpus/corpus.txt", "rt")
training_data = data.read()
sentences = open("../Corpus/sentences.txt", "rt")
testing_data = sentences.read()
bert_Kmeans_True = [3]*50 + [1]*50 + [2]*50 + [0]*50 

def get_datasest(text_list):
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = []
    for i, text in enumerate(text_list):
        ##如果是已经分好词的，不用再进行分词，直接按空格切分即可
        word_list = word_tokenize(text)
        #l = len(word_list)
        #word_list[l - 1] = word_list[l - 1].strip('')
        document = TaggededDocument(word_list, tags=[i])
        
        x_train.append(document)
    return x_train

def word_tokenizer(text):
        #tokenizes and stems the text
        tokens = word_tokenize(text)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
        return tokens

def cluster_sentences(training,testing, nb_of_clusters):
        tfidf_vectorizer = TfidfVectorizer()
        training_matrix = tfidf_vectorizer.fit_transform(training)
        #print(tfidf_matrix.toarray())
        #print("print feature")
        #print(tfidf_vectorizer.get_feature_names())
        kmeans = KMeans(n_clusters=nb_of_clusters,n_init = 20)
        kmeans.fit(training_matrix)
        testing_matrix = tfidf_vectorizer.fit_transform(testing)
                #print(tfidf_matrix.toarray())
                #print("print feature")
                #print(tfidf_vectorizer.get_feature_names())
        testing_label = kmeans.fit_predict(testing_matrix)
        score = silhouette_score(testing_matrix,testing_label)
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
            labels = KMeans(n_clusters=k).fit(testing_matrix).labels_
            score = metrics.silhouette_score(testing_matrix,labels)
            scores.append(score)
# 通过画图找出最合适的K值
        plt.plot(list(range(2,20)),scores)
        plt.xlabel('Number of Clusters Initialized')
        plt.ylabel('Sihouette Score')
        plt.show()

        return testing_label,score

if __name__ == "__main__":
        docs = training_data
    # 切成100句
        #text_list = docs.split("\n")
        a = docs.replace("? ?",".")
        b = a.replace('; ;','.')
        c = b.replace(':','.')
        train = nltk.sent_tokenize(c)
        test = nltk.sent_tokenize(testing_data)
        nclusters= 4
        labels,score = cluster_sentences(train, test,nclusters)
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
        tfidf_Kmeans_Res = [1, 0, 0, 3, 0, 0, 0, 3, 0, 1, 3, 0, 3, 0, 3, 3, 3, 3, 
                            3, 3, 3, 3,3, 2, 0, 0, 2, 1, 0, 0, 2, 0, 1, 1, 1, 3, 1, 1, 3, 0, 1, 1, 0, 3,
               3, 3, 0, 0, 2, 3, 0, 0, 1, 1, 0, 0, 1, 0, 3, 2, 1, 2, 0, 1, 3, 3,
               0, 1, 2, 2, 3, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 2, 2, 1, 0, 0, 0, 0, 3, 0, 1, 3, 2, 0, 0, 3, 2, 0, 3, 0, 3, 1,
               0, 2, 2, 0, 2, 0, 1, 3, 0, 2, 3, 2, 2, 3, 0, 2, 3, 2, 3, 3, 3, 3,
               3, 2, 0, 3, 3, 0, 3, 2, 3, 3, 3, 3, 0, 1, 3, 0, 0, 2, 0, 3, 1, 1,
               3, 2, 0, 3, 3, 0, 1, 0, 2, 3, 0, 3, 2, 3, 0, 2, 1, 1, 3, 0, 1, 0,
               2, 0, 1, 0, 0, 0, 3, 0, 0, 2, 2, 3, 3, 0, 0, 1, 1, 2, 0, 3, 0, 3,
               0, 1]

        tfidf_Kmeans_True = [1]*50 + [0]*50 + [3]*50 + [2]*50 

        f1_micro = f1_score(tfidf_Kmeans_True,tfidf_Kmeans_Res,average='micro')
        f1_macro = f1_score(tfidf_Kmeans_True,tfidf_Kmeans_Res,average='macro')
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
