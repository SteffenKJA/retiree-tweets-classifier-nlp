"""
Created on Mon Jun  4 17:32:31 2018

@author: Steffen_KJ
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from preprocessing_tools import preprocess_tweet

# Extract twitter data
dfMyData = pd.read_csv('dfMyData.csv')

tweets = dfMyData[u'vTweets']  # Raw tweets
target = dfMyData[u'target']  # Classification of tweets; normal or retired.

# Preprocess data
tweets = pd.Series([preprocess_tweet(tweet) for tweet in tweets])

use_same_ratio = True
ret_counter = 0

if use_same_ratio:
    selected_ret_tweets = target == 'Retired'
    normal_tweets = target == 'Normal'

    for x in range(len(selected_ret_tweets)):
        if ret_counter >= len(target[normal_tweets]):
            selected_ret_tweets[x] = False
        if selected_ret_tweets[x]:
            ret_counter += 1
    mask = selected_ret_tweets + normal_tweets
    tweets = tweets[mask]
    target = target[mask]

    assert len(target[target == 'Normal']) == len(target[target == 'Retired'])

# Do not use more than 2000 tokens in the vocabulary.
max_tokens = 2000
tokenizer = Tokenizer(num_words=max_tokens, split=' ')
tokenizer.fit_on_texts(tweets.values)
X = tokenizer.texts_to_sequences(tweets.values)
preproc_tweets = pad_sequences(X)

# The tokens are now arranged in a frequency matrix which can be read and
# understood by the neural network.

embed_dim = 128
lstm_out = 196  # Dimensionality of the output space

# The network is build as a series of sequential layers.
model = Sequential()

# Embedding takes (vocabulary length, dimensions where the words will be
# embedded, length of each word).
# Embedding is one layer in the network which has it own weights which will
# be trained and optimized.
model.add(Embedding(max_tokens, embed_dim,
                    input_length=preproc_tweets.shape[1]))

# Flatten the 2D output of Embedding to 1D, required by LSTM.
model.add(SpatialDropout1D(0.4))
# LSTM Long Short-Term Memory layer. A form of RNN (Recurrent Neural Network),
# where the past decisions of this layer are remembered, and affect the
# interpretation of input.

# Dropout is the fraction of neurons randomly deactivated in a layer for the
# fit - used to avoid overfitting, when the dataset appears too similar.
# Recurrent_dropout is the same as dropout, but for the recurrent (remembered)
# terms.
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

# In contrast to the dropout layer, a dense layer is simply a layer where
# each unit or neuron is connected to each neuron in the next layer.
model.add(Dense(2, activation='softmax'))

# Compile and optimize the neural network
#  - categorical_crossentropy is a suitable loss function for a categorical
#    classifier (as we have now).
#  - adam optimizer is an algorithm for first-order gradient-based optimization
#    of stochastic objective functions.

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

# Convert strings ('Normal', 'retired') to numerical values.
target_numerical = pd.get_dummies(target).values
X_train, X_test, Y_train, Y_test = train_test_split(preproc_tweets,
                                                    target_numerical,
                                                    test_size=0.3,
                                                    random_state=42)

# The size of the batch of tweets analyzed at each time. This saves time, when
# the dataset is very large.
batch_size = 32

# Train the model
model.fit(X_train, Y_train, epochs=40, batch_size=batch_size, verbose=2)

# Split the test set into a validation set for the model evaluation and one to count each
# correct/incorrect designation, i.e. a test set.
validation_size = int(len(X_test)/2.0)

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("score: %.2f" % (score))  # Evaluation of the mean loss
print("acc: %.2f" % (acc))  # Accuracy of predictions

retired_cnt, normal_cnt, retired_correct, normal_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    result = model.predict(X_validate[x].reshape(1, X_test.shape[1]),
                           batch_size=1, verbose=2)[0]

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

# --------------------------- A simple sanity check -------------------------- #

twt = ['The youth these days have no respect for elders.']
# vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
# padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=90, dtype='int32', value=0)

status = model.predict(twt, batch_size=1, verbose=2)[0]
if(np.argmax(status) == 0):
    print("Normal")
elif (np.argmax(status) == 1):
    print("Retired")

print('Done')
