
# coding: utf-8

# In[2]:


#C:\Users\Elliot\Packt\text databases\movies\aclImdb\train

# 1

from os import listdir
from os.path import isfile, join
mypath = 'movies/aclImdb/train/pos'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print('{} files in corpus'.format(len(onlyfiles)))

# 2

from nltk.tokenize import word_tokenize
text = [open(join(mypath, f), 'r', encoding='utf-8').readline() for f in onlyfiles[:50]]
text = [t for t in text if len(t) > 0]

print(text[2])
print(word_tokenize(text[2]))

# 3

allWords = [w for t in text for w in word_tokenize(t)]
print('{} total tokens in the reviews'.format(len(allWords)))

# 4

import pprint
from collections import Counter

wordCounts = Counter(allWords)
pprint.pprint(wordCounts.most_common(20))

# 5

allWords2 = [w.lower() for t in text for w in word_tokenize(t) if len(w) > 2]
print('{} total tokens in the reviews'.format(len(allWords2)))

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words()[660:690])

allStopWords = set(stopwords.words())
allUncommonWords = [w for w in allWords2 if w not in allStopWords]
print('{} total tokens in the reviews'.format(len(allUncommonWords)))
uncommonWordCounts = Counter(allUncommonWords)
pprint.pprint(uncommonWordCounts.most_common(20))

