#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:50:43 2018

@author: Steffen_KJ
"""

# This script creates a neural network to recognize breast cancer
from __future__ import division
from __future__ import print_function
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

cancer = load_breast_cancer()
#print(cancer.keys())

# Print full description by running:
#print(cancer['DESCR'])
# 569 data points with 30 features
cancer['data'].shape
#print(cancer)
X = cancer['data']
y = cancer['target']
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)


scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))

mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

#print(confusion_matrix(y_test, predictions))

#print(classification_report(y_test, predictions))

dfMyData = pd.read_csv('dfMyData.csv')
#dfMyDatavTweets = pd.read_csv('dfMyDatavTweets')

tweets = dfMyData[u'vTweets']
target = dfMyData[u'target']

# Split into train and target datasets
tweets_train, tweets_test, target_train, target_test = train_test_split(tweets,
                                                                        target)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(tweets_train)
X_train_counts.shape

# Just counting the number of words in each document has 1 issue: it will 
# give more weightage to longer documents than shorter documents. To avoid 
# this, we can use frequenc(TF - Term Frequencies) 
# i.e. #count(word) / #Total words, in each tweet.

# we can even reduce the weightage of more common words like 
# (the, is, an etc.) which occurs in all document. 
# TF-IDF i.e Term Frequency times inverse document frequency.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

clf = MultinomialNB().fit(X_train_tfidf, target_train)

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(fit_prior=True)),])

text_clf = text_clf.fit(tweets_train, target_train)

predicted = text_clf.predict(tweets_test)
print('Using naive algo, with stop_words')
print(np.mean(predicted == target_test))

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(tweets_train, target_train)

print("Using naive algo, no stem, res is")
print(gs_clf.best_score_)
print(gs_clf.best_params_)

import nltk
#nltk.download()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB(fit_prior=False)),])

text_mnb_stemmed = text_mnb_stemmed.fit(tweets_train, target_train)
predicted_mnb_stemmed = text_mnb_stemmed.predict(tweets_test)
print('With stem result is {}'.format(np.mean(predicted_mnb_stemmed == target_test)))

gs_clf_stem = GridSearchCV(text_mnb_stemmed, parameters, n_jobs=-1)
gs_clf_stem = gs_clf_stem.fit(tweets_train, target_train)

print("Using stem, res is:")
print(gs_clf_stem.best_score_)
print(gs_clf_stem.best_params_)

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, max_iter=5, random_state=42)),])
text_clf_svm = text_clf_svm.fit(tweets_train, target_train)
predicted_svm = text_clf_svm.predict(tweets_test)

print('With svm algo res is {}'.format(np.mean(predicted_svm == target_test)))

# =============================================================================
# parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
#                   'tfidf__use_idf': (True, False),
#                   'clf-svm__alpha': (1e-2, 1e-3),}
# gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
# gs_clf_svm = gs_clf_svm.fit(tweets_train, target_train)
# gs_clf_svm.best_score_
# gs_clf_svm.best_params_
# 
# =============================================================================


