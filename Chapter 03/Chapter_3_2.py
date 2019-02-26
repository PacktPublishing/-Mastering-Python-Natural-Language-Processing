import re
import pprint
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import treebank

WSJwords = treebank.tagged_words()
print('{} words read from WSJ treebank repository'.format(len(WSJwords)))

counts = Counter([x[1] for x in WSJwords])
pos_count = counts.most_common()
print('{} parts of speech in repository'.format(len(pos_count)))

pprint.pprint(pos_count)


plt.plot([np.log(x[1]) for x in pos_count])
plt.title('Parts of spech for WSJ words from Treebank repository')
plt.xlabel('rank of each part of speech')
plt.ylabel('number of occurrance for each part of speech')
plt.savefig(open("graphs/tagDistribution.png", 'w')
plt.show()


allPOS = [x[0] for x in pos_count]

selectedWords = WSJwords
keepPOS = allPOS
wDict = dict((w,i) for i,w in enumerate(keepPOS))

import pandas as pd

def getTwoGram(keepPOS, words):
    twoGram = np.zeros((len(keepPOS), len(keepPOS)))
    word1 = wDict[words[0][1]]
    for i in range(1,len(words)):
        word2 = wDict[words[i][1]]
        twoGram[word1, word2] += 1
        word1 = word2
    return pd.DataFrame(twoGram, index=keepPOS, columns=keepPOS)

twoGram = getTwoGram(keepPOS, selectedWords)
np.array(twoGram)[:5,:5]

# create conditionly probability matrix for tags: close match to Jurafsky p. 163 table

allTwoGramPct = (twoGram.T / twoGram.sum(axis=1)).T
np.array(allTwoGramPct)[:5,:5]
allTwoGramPct_pd = pd.DataFrame(np.array(allTwoGramPct), columns=keepPOS, index=keepPOS)
allTwoGramPct_pd.iloc[:5,:5]

# create conditional probability for likelihood for each tag to each word

sentence = 'Bill and John can and should both share first place'
wordList =[(x,1) for x in sentence.split(' ')]
uniqueWords = list(set(wordList))

matchCount = [(wc, Counter([x[1] for x in selectedWords if x[0] == wc[0]]))  for wc in uniqueWords]
pprint.pprint(matchCount)

tagSet = set([x for sublist in [list(x[1]) for x in matchCount] for x in sublist])
tagDict = dict([(x,i) for i,x in enumerate(tagSet)])
tagSet

tagLikelihood = np.zeros((len(tagSet),len(matchCount)))
for i, w in enumerate(matchCount):
    for like in list(w[1].items()):
        tagLikelihood[tagDict[like[0]], i] = like[1]

tagLike_pd = pd.DataFrame(np.array(tagLikelihood), columns=[x[0][0] for x in matchCount], index=list(tagSet))  
tagLike_pd
 
tagLikePct = (tagLike_pd.T/tagLike_pd.sum(axis=1)).T
tagLikePct

selectTwoGramPct = allTwoGramPct_pd.loc[tagSet, tagSet]
selectTwoGramPct

# convert to json for input into the Viterbi algorithm

def convertToDict(tagLikePct):
    colDict = dict([(i,x) for i,x in enumerate(list(tagLikePct.columns))])
    rowList = list(tagLikePct.index)
    xx = tagLikePct.transpose()
    return dict([(rowList[i],dict([(colDict[i],x) for i,x in enumerate(xx.iloc[:,i])])) for i in range(len(rowList))])

startCount = allTwoGramPct.transpose()['.'].transpose() 

obs = tuple([x[0] for x in wordList])
states = tuple(list(selectTwoGramPct.columns))
start_p = {k:v for k,v in dict(startCount).items() if k in states}
trans_p = convertToDict(selectTwoGramPct)
obs_p = convertToDict(tagLikePct)

# apply the Vitberi algorithm

V, opt = viterbi(obs, states, start_p, trans_p, obs_p)
