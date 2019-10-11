# Sentiment Analysis: approximating Word2Vec results with SPPMI-SVD
## The project
Introduced by Mikolov et al. in two papers in 2013 (Mikolov et al. [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/pdf/1301.3781.pdf) and Mikolov et al. [*Distributed Representations of Words and Phrases and their Compositionality*](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) ), Word2Vec is a widely popular way of getting vector representation for words. The core assumption of Word2Vec is that if two words are used in similar contexts, then they should share a similar meaning and vector representation. These vector representations can then be used in clustering a set of documents or in text classification tasks.

In 2014, Omer Levy and Yoav Goldberg demonstrated that Word2Vec could be approximated by "factorizing a word-context matrix whose cells are the pointwise mutual information (PMI) of the respective word and context pairs" ([*Neural Word Embedding as Implicit Matrix Factorization*](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)). Unlike Word2Vec which is based on a neural-network and uses gradient descent, Levy and Goldberg method only relies on word count, information theory and the factorization of a matrix with the well-known Singular Value Decomposition. They also go on and show that this method produces word embeddings that can achieve comparable performance as the ones from Word2Vec.

This project compares the performance of [Gensim's](https://radimrehurek.com/gensim/models/word2vec.html) famous implementation of Word2Vec and my own implementation of Levy and Goldberg model on a Sentiment Analysis tasks.

## The Data
The data comes from Ahmed Besbes *Sentiment Analysis on twitter using word2vec and keras* [blog post](https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html). You can find the data [here](https://drive.google.com/uc?id=0B04GJPshIjmPRnZManQwWEdTZjg&export=download).

The data consists of more than a million of tweets, each tweet comes with its content / text and a binary sentiment score (positive / negative feeling).

## The Code
The code is organized as follow: a light pre-processing of the data, the creation of both a Word2Vec and a SPPMI - SVD models, the training of a classifier for each resulting word-embeddings, the comparison of the two resulting sentiment classification. All in a Python Jupyter Notebook.

### Part 1. Light Pre-processing of the data
A brief pre-processing of the tweets, removing hyperlinks and hashtags, and keeping tweets long enough to learn word-embeddings from.

### Part 2.  Creation of both a Word2Vec and a SPPMI - SVD models
Using Gensim's Word2Vec implementation, and my own for SPPMI - SVD, built two differents vector representations of the corpus' words. With these vector representations, transformed each tweet into a vector using [spaCy](https://spacy.io/).

### Part 3. Training of a classifier for each resulting word-embeddings
Trained two [Stochastic Gradient Descent classifiers](https://scikit-learn.org/stable/modules/sgd.html), one for each word-embeddings.

### Part 4. Comparison of the two resulting sentiment classification
Compared different metrics of the sentiment classification tasks between the two classifiers.

## The Results
Obtained decent sentiment classification performance from both Word2Vec and SPPMI-SVD. Even better, SPPMI - SVD provided results very close to Word2Ve, providing a decent alternative to Word2Vec.
