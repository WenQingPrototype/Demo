# get brown corpus from nltk
from nltk.corpus import brown


#print(brown.categories())



T1 = brown.sents(categories= "science_fiction")
T2 = brown.sents(categories= "romance")[:1000]
T3 = brown.sents(categories= "mystery")[:1000]
T4 = brown.sents(categories= "humor")
T5 = brown.sents(categories= "government")[:1000]


#print(len(T1))
#print(len(T2))
#print(len(T3))
#print(len(T4))
#print(len(T5))
#print(T1)

labels = ['science_fiction', 'romance', 'mystery', 'humor', 'government']




t1 = ""
for i in T1:
	text = ' '.join(i)
	t1  = t1 + text
t2 = ""
for i in T2:
	text = ' '.join(i)
	t2  = t2 + text
t3 = ""
for i in T3:
	text = ' '.join(i)
	t3  = t3 + text
t4 = ""
for i in T4:
	text = ' '.join(i)
	t4  = t4 + text
t5 = ""
for i in T5:
	text = ' '.join(i)
	t5  = t5 + text
text = t1+t2+t3+t4+t5

#print(len(text))
##words = stopwords.words('english')
#print(words)
#for w in ['!',',','.','?','-s','-ly','</s>','s']:
#    words.add(w)


#output the corpus in txt file form
for category in brown.categories():
	sentences = brown.sents(categories=category)
	print(len(sentences), "sentences in", category)
	text = "\n".join([" ".join(s) for s in sentences])
	filename = category + '.txt'
	outfile = open(filename, 'w')
	outfile.write(text)
	outfile.close()