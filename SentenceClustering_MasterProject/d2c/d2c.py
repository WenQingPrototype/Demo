from nltk import word_tokenize     # coding:utf-8
import gensim
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
#from t2 import text


data = open("../Corpus/corpus.txt", "rt")
training_data = data.read()

sentences = open("../Corpus/sentences.txt", "rt")
testing_data = sentences.read()

def get_datasest(text_list):
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = []
    for i, text in enumerate(text_list):

        word_list = word_tokenize(text)
        l = len(word_list)
        #word_list[l - 1] = word_list[l - 1].strip('')
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train

def train(x_train, size=200, epoch_num=1): ##size 是你最终训练出的句子向量的维度，自己尝试着修改一下

    model_dm = Doc2Vec(x_train, min_count=1, window=5, vector_size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('../model/model.w2c') ##模型保存的位置

    return model_dm


def test(test_data):
    model_dm = Doc2Vec.load("../model/model.w2c")

    str1 = test_data

    test_text = word_tokenize(str1)
    #print(test_text)
    inferred_vector_dm = model_dm.infer_vector(test_text) ##得到文本的向量
    print(inferred_vector_dm)

    return inferred_vector_dm


if __name__ == '__main__':
    # TRAIN--------------------------------
    
    docs = training_data
    # 切成100句
    text_list = docs.split("\n")
    # 去除第一个的回车
    #text_list[0] = text_list[0][1:]
    x_train = get_datasest(text_list)
    model_dm = train(x_train)
    # GET VECTORS ----------------------------
    doc_2_vec = test(testing_data)
    #print(type(doc_2_vec))
    print(doc_2_vec.shape)
    import pandas as pd 


    pd.DataFrame(doc_2_vec).to_csv('../model/d2c_array.csv')

