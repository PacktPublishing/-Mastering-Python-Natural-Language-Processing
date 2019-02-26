
# coding: utf-8

# In[3]:



# 1

from os import listdir
from os.path import isfile, join
import pandas as pd

def readText(mypath, n=50):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    text = [open(join(mypath, f), 'r', encoding='utf-8').readline() for f in onlyfiles[:n]]
    return ([t for t in text if len(t) > 0])

positive = readText('movies/aclImdb/train/pos', 500)
negative = readText('movies/aclImdb/train/neg', 500)

positive_pd = pd.DataFrame(list(zip(positive, [1] * len(positive))), columns=['text','pos_neg'])
negative_pd = pd.DataFrame(list(zip(negative, [0] * len(positive))), columns=['text','pos_neg'])
result = pd.concat([positive_pd, negative_pd])

# 2

from sklearn.feature_extraction.text import TfidfVectorizer	
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import math

class TextToTfidfMatrix:
    
    def __init__(self, features_train, min_df=5):
        self.count_vect = CountVectorizer(min_df=min_df , max_df=.95, stop_words='english')
        self.tfidfTransform = TfidfTransformer()
        X_train_counts = self.count_vect.fit_transform(features_train)
        self.X_train = self.tfidfTransform.fit_transform(X_train_counts)
    
    def trainingTextMatrix(self):
        return self.X_train
    
    def textToTfidfMatrix(self, features_test):
        X_test_counts = self.count_vect.transform(features_test)
        X_test = self.tfidfTransform.transform(X_test_counts)
        return X_test
    
    def trainAndTestMatrices(self, features_test):
        return self.X_train, self.textToTfidfMatrix(features_test) 
    
    def tfidf(self):
        return self.tfidfTransform
    
    def vocabulary(self):
        return self.count_vect.vocabulary_

featuresSelect = result['text']
labelsSelect = result['pos_neg']
features_train, features_test, labels_train, labels_test = train_test_split(featuresSelect, labelsSelect, test_size=0.4, random_state=0)

select = TextToTfidfMatrix(features_train)
print(len(select.vocabulary()))
print(list(select.vocabulary().keys())[:5])
print(select.trainingTextMatrix())

# 3

X_train, X_test = select.trainAndTestMatrices(features_test)
y_train, y_test = [labels_train, labels_test]

clfLR = LogisticRegression(C=2, penalty='l1').fit(X_train, y_train) 
predicted = clfLR.predict(X_test)

confusion = metrics.confusion_matrix(y_test, predicted)
print(confusion)

# 4

def getWordWeights(clf, select):
    vocab = dict([(c,x) for (x,c) in select.count_vect.vocabulary_.items()])
    nonZeroWords = [(c,x, vocab[x]) for (x,c) in enumerate(list(clf.coef_[0])) if c != 0]
    return sorted(nonZeroWords, key=lambda x: x[0])

def showMostDiagnosticWords(select, clf):
    wordWeights = getWordWeights(clf, select)
    listLength = math.floor(max(1,min(10,1+len(wordWeights)/2)))
    
    print(pd.concat([pd.DataFrame(wordWeights[:listLength], index=range(listLength), columns=['bottom coef','id','word']),
               pd.DataFrame(wordWeights[-listLength:], index=range(listLength), columns=['top coef','id','word'])], axis=1))


showMostDiagnosticWords(select, clfLR)

