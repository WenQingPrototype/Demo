import transformers
from transformers import BertTokenizer
from transformers import BertModel
from transformers import logging

logging.set_verbosity_warning()

data = open("../Corpus/sentences.txt", "rt")
text = data.read()
sentences = text.split(".\n\n")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")



encoded_input3 = tokenizer(sentences, padding=True, return_tensors="pt")
bert_output3 = model(input_ids=encoded_input3['input_ids'])
# sentence embedding
test_data = bert_output3.last_hidden_state[:,0,:]



import torch
import numpy as np
print("tensor:\n",test_data) 
vector_bert = test_data.detach().numpy()
#vector_bert.save('../model/bert_vector.txt')


# import csv

# 1. create file object
f = open('../model/bert_array.csv', 'w', encoding='utf-8')

# 2. 基于文件对象构建 csv写入对象Building csv write objects based on file objects
#csv_writer = csv.writer(f)

#row = []
#for result in vector_bert:
#    row.append(result.get_value())

# 3. 构建列表头
#with open(output_path, "wb") as file:
   # writer = csv.writer(file)
  #  writer.writerow(row)

# 4. 写入csv文件内容
import pandas as pd 


pd.DataFrame(vector_bert).to_csv('../model/bert_array.csv')
# 5. 关闭文件
#f.close()

#np.savetxt("../model/bert_vector.txt", vector_bert,fmt='%f',delimiter=',')