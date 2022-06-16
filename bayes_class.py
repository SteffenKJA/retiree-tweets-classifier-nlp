#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer

from NN_mods import preprocess_tweet

# Handles

use_nlp_algo = True
use_same_ratio = True

class StemmedCountVectorizer(CountVectorizer):
    """
    This class counts and returns a list of stemmed words, from a given tweet.
    """
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


# ========================= EXTRACT AND PREPARE DATA ==========================

# Extract twitter data
dfMyData = pd.read_csv('dfMyData.csv')

tweets = dfMyData[u'vTweets']  # Raw tweets
target = dfMyData[u'target']  # Classification of tweets; normal or retired.
ret_counter = 0

if use_same_ratio:
    selected_ret_tweets = target == 'Retired'
    normal_tweets = target == 'Normal'

    for x in xrange(len(selected_ret_tweets)):
        if ret_counter >= len(target[normal_tweets]):
            selected_ret_tweets[x] = False
        if selected_ret_tweets[x]:
            ret_counter += 1
            print(ret_counter)
    mask = selected_ret_tweets + normal_tweets

    tweets = tweets[mask]
    target = target[mask]

    print('len is', len(target[target == 'Normal']))
    print('len is', len(target[target == 'Retired']))

    assert len(target[target == 'Normal']) == len(target[target == 'Retired'])

if use_nlp_algo:
    tweets = [preprocess_tweet(tweet) for tweet in tweets]
    print(type(tweets))

    # Split data into training and test data. 80 % of the total data is used
    # as a training dataset.
    tweets_train, tweets_test, target_train, target_test = train_test_split(tweets,
                                                            target, test_size=0.2,
                                                            random_state=2321)

    # ====================== SETUP TOKENAZATION OF TWEETS =====================

    # CountVectorizer converts a given list of words into a matrix of
    # token counts.
    count_vect = CountVectorizer()
    print('count_vect is', count_vect)
    # Build a vocabulary of words based on the input tweets.
    tweets_train_counts = count_vect.fit_transform(tweets_train)

    print(tweets_train_counts)
    # Just counting the number of words in each tweet has 1 issue: it will
    # give a higher weight to longer documents as opposed to shorter documents.
    # To avoid this, we can use frequency TF (Term Frequencies)
    # i.e. count(word) / total_words, in each tweet.

    # we can even reduce the weightage of more common words like
    # (the, is, an etc.) which occurs in all tweets.
    # TF-IDF i.e Term Frequency times the inverse frequency in all tweets.

    # So, lets transform the count matrix to a normalized tf or tf-idf
    # representation
    tfidf_transformer = TfidfTransformer()
    tweets_train_tfidf = tfidf_transformer.fit_transform(tweets_train_counts)

    # ============================ CLASSIFIER =================================

    # Build and train the final classifier (clf)
    clf = MultinomialNB().fit(tweets_train_tfidf, target_train)
    print(clf.get_params().keys())

    # ============================= PIPELINE ==================================

    # Create pipeline for all the transformations. Some notes;
    # stop words are common words (they, that, this etc) which are filtered
    # out.
    # Pipeline recreates the above steps tersely in one line of code.
    # fit_prior is whether or not to learn class prior probabilities, i.e,
    # the prior likelihood of a retired or normal tweet.
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB(fit_prior=True))])

    # Train our neural network.
    text_clf = text_clf.fit(tweets_train, target_train)

    # Get our estimates and validate result.
    predicted = text_clf.predict(tweets_test)
    print('Using naive Bayes, no optimization')
    print(np.mean(predicted == target_test))

    # =========================== OPTIMIZATION ================================

    # The alpha is a smoothing parameter, dealing with null likelihoods.
    # E.g., a word used in a tweet that was not in the dataset would result in
    # a zero-division. Alpha is then a small offset value.
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1.0, 1e-10)}

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=8)
    gs_clf = gs_clf.fit(tweets_train, target_train)

    print("Using naive Bayes, no stem, with optimization, has a validity of")
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    retired_cnt, normal_cnt, retired_correct, normal_correct = 0, 0, 0, 0
    result = gs_clf.predict(tweets_test)
# =============================================================================

    for res, target in zip(result, target_test):
        if res == target:
            if res == 'Normal':
                normal_correct += 1
            else:
                retired_correct += 1

        if target == 'Normal':
            normal_cnt += 1
        else:
            retired_cnt += 1

    print("retired_acc", retired_correct/retired_cnt*100, "%")
    print("normal_acc", normal_correct/normal_cnt*100, "%")

    # =========================== USING STEMMED WORDS =========================

    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

    text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', MultinomialNB(fit_prior=False))])

    text_mnb_stemmed = text_mnb_stemmed.fit(tweets_train, target_train)
    predicted_mnb_stemmed = text_mnb_stemmed.predict(tweets_test)
    print('No optimization, with stem, has a validity of {}'.format(
                                    np.mean(predicted_mnb_stemmed == target_test)))

    gs_clf_stem = GridSearchCV(text_mnb_stemmed, parameters, n_jobs=8)
    gs_clf_stem = gs_clf_stem.fit(tweets_train, target_train)

    print("Optimization, with stem, has a validity of:")
    print(gs_clf_stem.best_score_)
    print(gs_clf_stem.best_params_)

    print('Done.')
