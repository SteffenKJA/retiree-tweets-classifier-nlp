#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import re
import urlparse
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import nltk
# nltk.download()
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
import json
from keras.preprocessing.sequence import pad_sequences

import datetime
import time

stemmer = LancasterStemmer()
# Handles

use_nlp_algo = True
use_NN = False
preproc_NN_data = False
use_NN_py = False

class StemmedCountVectorizer(CountVectorizer):
    """
    This class counts and returns a list of stemmed words, from a given tweet.
    """
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def preprocess_word(word):
    """
    Credit of abdulfatir (github).
    """
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    """
    Credit of abdulfatir (github).
    """
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    """
    Credit of abdulfatir (github).
    """
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    """
    Credit of abdulfatir (github).
    """

    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)

    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    #words = tweet.split()
    # Remove recurring characters in words
    re.sub(r'(.)\1+', r'\1\1', tweet)
    #print(tweet)
    
    #for word in words:
    #    word = preprocess_word(word)
    #    if is_valid_word(word):
            #if use_stemmer:
            #    word = str(porter_stemmer.stem(word))
    #        processed_tweet.append(word)

    #return ' '.join(processed_tweet)
    return tweet


# =============================================================================
#                           Neural network functions
# =============================================================================

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

# =============================================================================
#                    Main function for neural network
# =============================================================================
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False,
          dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons,
                                                                  str(alpha),
                                                                  dropout,
                                                                  dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]),
                                                             1, len(classes)))
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        print('layer_0 is', layer_0)
        print('synapse_0 is', synapse_0)
        print(np.dot(layer_0, synapse_0))
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))],
                                           1 - dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration,
            # break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" +
                       str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error
        # (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
               }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)

def classify(sentence, ERROR_THRESHOLD=0.2, show_details=False):
    results = think(sentence, show_details)

    results = [[i, r] for i, r in enumerate(results) if r >
               ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results =[[classes[r[0]], r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results


# ========================= EXTRACT AND PREPARE DATA ==========================

# Extract twitter data
dfMyData = pd.read_csv('dfMyData.csv')

tweets = dfMyData[u'vTweets']  # Raw tweets
target = dfMyData[u'target']  # Classification of tweets; normal or retired.

print(type(tweets))

# tweets = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", x).split()) for x in tweets]
#tweets = [re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE) for tweet in tweets]
#tweets = [re.sub(r'http\S+', '', tweet, flags=re.MULTILINE) for tweet in tweets]

if use_nlp_algo:
    tweets = [preprocess_tweet(tweet) for tweet in tweets]
    print(type(tweets))

    # =============================================================================
    # new_tweet = ''
    # # Remove urls
    # processed_tweets = []
    # for tweet in tweets:
    #     for i in tweet.split():
    #         s, n, p, pa, q, f = urlparse.urlparse(i)
    #         if s and n:
    #             pass
    #         elif i[:1] == '@':
    #             pass
    #         elif i[:1] == '#':
    #             new_tweet = new_tweet.strip() + ' ' + i[1:]
    #         else:
    #             new_tweet = new_tweet.strip() + ' ' + i
    #     processed_tweets.append(new_tweet)
    # =============================================================================

    # Split data into training and test data. 85 % of the total data is used
    # as a training dataset.
    tweets_train, tweets_test, target_train, target_test = train_test_split(tweets,
                                                            target, test_size=0.2,
                                                            random_state=2321)
    
    # ====================== SETUP TOKENAZATION OF TWEETS =========================
    
    # CountVectorizer converts a given list of words into a matrix of token counts.
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

    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1.0, 1e-10)}

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=8)
    gs_clf = gs_clf.fit(tweets_train, target_train)

    print("Using naive Bayes, no stem, with optimization, has a validity of")
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    # Convert strings ('Normal', 'retired') to numerical values.
    target_numerical = pd.get_dummies(target).values
    tweets_train, tweets_test, target_train, target_test = train_test_split(
                                                            tweets,
                                                            target_numerical,
                                                            test_size=0.2,
                                                            random_state=2321)

    # Split the test set into one for the model evaluation and one to count
    # each correct/incorrect designation.
    validation_size = int(len(tweets_test)/2.0)
    print('Length of validation set is {}'.format(validation_size))

    X_validate = tweets_test[-validation_size:]
    Y_validate = target_test[-validation_size:]
    X_test = tweets_test[:-validation_size]
    Y_test = target_test[:-validation_size]
    retired_cnt, normal_cnt, retired_correct, normal_correct = 0, 0, 0, 0
    result = gs_clf.predict(X_validate)
    print('result is', result)
    for x in range(len(X_validate)):
#        result = gs_clf.predict(X_validate[x].reshape(1, X_test.shape[1]),
#                                batch_size=1, verbose=2)[0]
        
        if np.argmax(result) == np.argmax(Y_validate[x]):
            if np.argmax(Y_validate[x]) == 0:
                normal_correct += 1
            else:
                retired_correct += 1

        if np.argmax(Y_validate[x]) == 0:
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
    

# =============================================================================
# # Neural Network
# =============================================================================

elif use_NN:
    # Prepare data into words and classes
    # Split data into training and test data. 85 % of the total data is used
    # as a training dataset.
    tweets_train, tweets_test, target_train, target_test = train_test_split(tweets,
                                                            target, test_size=0.9,
                                                            random_state=2321)
    words = []
    classes = []
    documents = []
    ignore_words = ['?']

    # loop through each sentence in our training data

    for tweet, target in zip(tweets_train, target_train):
        # tokenize each word in the sentence
        w = nltk.word_tokenize(tweet)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, target))
        # add to our classes list
        if target not in classes:
            classes.append(target)

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(word.lower()) for word in words if word not in
             ignore_words]
    words = list(set(words))

    # remove duplicates
    classes = list(set(classes))

    print (len(documents), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique stemmed words", words)

    # Train the data.
    if preproc_NN_data:
        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(classes)

        # training set, bag of words for each sentence
        count = 0
        #print('words are', words)
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            #print('pattern words before')
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in
                             pattern_words]
            #print('pattern words after')

            # create our bag of words array
            for w in words:
               # print('w is', w)
                bag.append(1) if w in pattern_words else bag.append(0)
            #print('bag', bag)
            training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            output.append(output_row)
            #print('output', output)
            count += 1

            if count in range(5000, 130000, 5000):
                print('Now at tweet number {}'.format(count))

        # sample training/output
        i = 0
        w = documents[i][0]
        print([stemmer.stem(word.lower()) for word in w])
        print(training[i])
        print(output[i])

        df_training = pd.Series(training)
        df_output = pd.Series(output)
        os.chdir('/Users/Steffen_KJ/Dropbox/Nets')
        df_training.to_csv('training_NN.csv')
        df_output.to_csv('output_NN.csv')

    # Extract pre-processed data
    training = pd.read_csv('training_NN.csv')
    output = pd.read_csv('output_NN.csv')
    print(type(training))
    X = np.array(training)
    y = np.array(output)
    print(type(X))
    start_time = time.time()

    train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False,
          dropout_percent=0.2)

    elapsed_time = time.time() - start_time
    print ("processing time:", elapsed_time, "seconds")

    # probability threshold
    ERROR_THRESHOLD = 0.2
    # load our calculated synapse values
    synapse_file = 'synapses.json'

    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])

    print(classify("sudo make me a sandwich"))

elif use_NN_py:
    import numpy as np

    # extract data from a csv
    # notice the cool options to skip lines at the beginning
    # and to only take data from certain columns
    # training = np.genfromtxt('path/to/your/data.csv', delimiter=',',
    #                         skip_header=1, usecols=(1, 3), dtype=None)
    tweets_train, tweets_test, target_train, target_test = train_test_split(tweets,
                                                            target, test_size=0.2,
                                                            random_state=2321)

    # create our training data from the tweets
    train_x = np.asarray([x for x in tweets_train])
    # index all the sentiment labels
    train_y = []
    print(target_train)
    for x in target_train:
        if x == 'Retired':
            train_y.append(1)
        elif x == 'Normal':
            train_y.append(0)
    train_y = np.asarray([train_y])

    import json
    import keras
    import keras.preprocessing.text as kpt
    from keras.preprocessing.text import Tokenizer

    # only work with the 3000 most popular words found in our dataset
    max_words = 3000

    # create a new Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    # feed our tweets to the Tokenizer
    tokenizer.fit_on_texts(train_x)

    # Tokenizers come with a convenient list of words and IDs
    dictionary = tokenizer.word_index
    # Let's save this out so we can use it later
    with open('dictionary.json', 'w') as dictionary_file:
        json.dump(dictionary, dictionary_file)

    def convert_text_to_index_array(text):
        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

    allWordIndices = []
    # for each tweet, change each token to its ID in the Tokenizer's word_index
    for text in train_x:
        wordIndices = convert_text_to_index_array(text)
        allWordIndices.append(wordIndices)

    # now we have a list of all tweets converted to index arrays.
    # cast as an array for future usage.
    allWordIndices = np.asarray(allWordIndices)

    # create one-hot matrices out of the indexed tweets
    X = tokenizer.texts_to_sequences(train_x)
    X = pad_sequences(X)
    train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    # treat the labels as categories
    train_y = keras.utils.to_categorical(train_y, 2)

    from keras.models import Sequential
    from keras.layers import (Dense, Dropout, Activation, Flatten, LSTM,
                              Embedding, SpatialDropout1D)

    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_words, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
#    model.add(Dense(512, input_shape=(max_words,), activation='relu'))
   # model.add(Dropout(0.5))
   # model.add(Dense(256, activation='sigmoid'))
   # model.add(Dropout(0.5))
   # model.add(Dense(1, activation='softmax'))
  #  model.add(Flatten())
   # model.add(Dense(output_dim=2,init ='uniform', activation = 'relu'))
    print('Adding done')
   # model.compile(loss='categorical_crossentropy',
   #               optimizer='adam',
   #               metrics=['accuracy'])
    print('Compile done')
    batch_size = 32
    model.fit(tweets_train, target_train, epochs=7, batch_size=batch_size,
              verbose=2)

    #model.fit(train_x, train_y,
    #          batch_size=32,
    #          epochs=5,
    #          verbose=1,
    #          validation_split=0.1,
    #          shuffle=True)

# ============================= SVM ALGO ======================================

# =============================================================================
# from sklearn.linear_model import SGDClassifier
# text_clf_svm = Pipeline([('vect', CountVectorizer()),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf-svm', SGDClassifier(loss='perceptron',
#                                                    penalty='elasticnet',
#                          alpha=1e-3, max_iter=25, random_state=42)),])
# text_clf_svm = text_clf_svm.fit(tweets_train, target_train)
# predicted_svm = text_clf_svm.predict(tweets_test)
#
# print('With svm algo res is {}'.format(np.mean(predicted_svm == target_test)))
# =============================================================================

# =============================================================================
# parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
#                   'tfidf__use_idf': (True, False),
#                   'clf-svm__alpha': (1e-2, 1e-3),}
# gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=8)
# gs_clf_svm = gs_clf_svm.fit(tweets_train, target_train)
# gs_clf_svm.best_score_
# gs_clf_svm.best_params_
#
# =============================================================================

