
# 1

from os import listdir
from os.path import isfile, join
import pandas as pd

def readText(mypath, n=50):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    text = [open(join(mypath, f), 'r', encoding='utf-8').readline() for f in onlyfiles[:n]]
    return ([t for t in text if len(t) > 0])

ndocs = 20000

positive = readText('movies/aclImdb/train/pos', ndocs)
negative = readText('movies/aclImdb/train/neg', ndocs)

positive_pd = pd.DataFrame(list(zip(positive, [1] * len(positive))), columns=['text','pos_neg'])
negative_pd = pd.DataFrame(list(zip(negative, [0] * len(positive))), columns=['text','pos_neg'])
result = pd.concat([positive_pd, negative_pd])

# 2

from sklearn.feature_extraction.text import TfidfVectorizer	
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

minpct = .005
min_df = int(features_train.shape[0] * minpct)
print('words appear in at least {} documents'.format(min_df))

select = TextToTfidfMatrix(features_train, min_df)

X_train, X_test = select.trainAndTestMatrices(features_test)
y_train, y_test = [labels_train, labels_test]

#3 

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from keras.callbacks import History 

import pprint, json

num_classes = 2
num_terms = X_train.shape[1]

x_train_k = X_train.toarray() #.reshape(X_train_array.shape[0], num_terms) #, 1)
x_test_k = X_test.toarray() #.reshape(X_test_array.shape[0], num_terms) #, 1)
input_shape = (num_terms, )

x_train_k = x_train_k.astype('float32')
x_test_k = x_test_k.astype('float32')

y_train_k = keras.utils.to_categorical(y_train, num_classes)  # one-hot encoding
y_test_k = keras.utils.to_categorical(y_test, num_classes)


#4

neurons_per_layer = [4,8] #[64, 32]

model = Sequential()
model.add(Dense(neurons_per_layer[0], activation='relu', input_shape=input_shape))
model.add(Dense(neurons_per_layer[1], activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(num_classes, activation='softmax'))  

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
epochs = 50
history = History()

model.fit(x_train_k, y_train_k,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(x_test_k, y_test_k), callbacks=[history])


loss, accuracy = model.evaluate(x_test_k, y_test_k, verbose=0)
print('Test cross-entropy value = {}\nTest percent accuracy = {}'.
      format(loss, accuracy))

train_loss, train_accuracy = model.evaluate(x_train_k, y_train_k, verbose=0)
print('Training set  cross-entropy value = {}\nTraining set percent accuracy = {}'.
      format(train_loss, train_accuracy))

# 5

import matplotlib.pyplot as plt
plt.clf()

historyLabels = ['test cross-entropy','test accuracy','train cross-entropy','train accuracy']
plotLabelDict = dict(np.array([list(history.history.keys()),historyLabels]).transpose())

for i,stat in enumerate(list(history.history.keys())):
    plt.subplot(2,2,i+1)
    plt.plot(history.history[stat])
    plt.ylabel(plotLabelDict[stat])
    plt.xlabel('iterations')

plt.tight_layout()
plt.show()
#plt.savefig('graphics/Chapter2 NN performance 4x8.png')

#6 

def displayFitStatistics(y_test01, predicted01):
    print('f1 = {0:0.3}; recall = {1:0.3}; precision = {2:0.3}'.format(
        metrics.f1_score(y_test01, predicted01), 
        metrics.recall_score(y_test01, predicted01), 
        metrics.precision_score(y_test01, predicted01)))

    confusion = metrics.confusion_matrix(y_test01, predicted01)
    return confusion

predicted = model.predict(x_test_k)
predicted01 = [np.argmax(x) for x in predicted]

print(displayFitStatistics(y_test, predicted01))

'''# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to JSON
model_json = model.to_json()
with open("chapter2_1_tf.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("chapter2_1_tf.h5")
'''
model.save('chapter2_1_tf_model.h5')

# https://keras.io/models/about-keras-models/
model.get_weights()

pprint.pprint([x.shape for x in model.get_weights()])
pprint.pprint(json.loads(model_json))

model.summary()

from keras.models import load_model
model1 = load_model('chapter2_1_tf_model.h5')

predicted = model1.predict(x_test_k)
predicted01 = [np.argmax(x) for x in predicted]
y_test01 = [np.argmax(x) for x in y_test_k]

print(displayFitStatistics(y_test01, predicted01))

help(model)
