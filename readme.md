Case study of twitter tweets classification. Is the author retired (senior citizen) or not?

A simple neural network in Keras is used, as well as a Naive Bayes classifier. If we add a third model we could
in principle create a ensemble classification model.

Public prezi presentation can be found at https://prezi.com/view/gCuzrfYFexUcNCIuCBtX/

stop_words='english' did not have much impact. Grid search pars still show 79 %. 

Links:
https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-11-cnn-word2vec-41f5e28eda74
http://scikit-learn.org/stable/modules/neural_networks_supervised.html

A reference for the classification metrics on this twitter case, is suicide indicators from tweets, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4886102/;
"Our findings show that people who are at high suicidal risk can be easily differentiated from those who are not by machine learning algorithms, which accurately identify the clinically significant suicidal rate in 92% of cases (sensitivity: 53%, specificity: 97%, positive predictive value: 75%, negative predictive value: 93%"
