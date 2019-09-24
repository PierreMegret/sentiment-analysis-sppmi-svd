# Sentiment Analysis: approximating Word2Vec results with SPPMI-SVD
## The project
Introduced by Mikolov et al. in two papers in 2013 (Mikolov et al. [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/pdf/1301.3781.pdf) and Mikolov et al. [*Distributed Representations of Words and Phrases and their Compositionality*](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) ), Word2Vec is a widely popular way of getting vector representation for words. The core assumption of Word2Vec is that if two words are used in similar contexts, then they should share a similar meaning and vector representation. These vector representations can then be used in clustering a set of documents or in text classification tasks.

In 2014, Omer Levy and Yoav Goldberg demonstrated that Word2Vec could be approximated by "factorizing a word-context matrix whose cells are the pointwise mutual information (PMI) of the respective word and context pairs" ([*Neural Word Embedding as Implicit Matrix Factorization*](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)). Unlike Word2Vec which is based on a neural-network and uses gradient descent, Omer Levy and Yoav Goldberg method only relies on word count, information theory and the factorization of a matrix with the well-known Singular Value Decomposition. They also go on and show that this method produces word embeddings that can achieve comparable performance as the ones from Word2Vec.

This project compares the performance of [Gensim's](https://radimrehurek.com/gensim/models/word2vec.html) famous implementation of Word2Vec and my own implementation of Omer Levy and Yoav Goldberg model on a Sentiment Analysis tasks.

## The Data
The data comes from Ahmed Besbes *Sentiment Analysis on twitter using word2vec and keras* [blog post](https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html). You can find the data [here](https://drive.google.com/uc?id=0B04GJPshIjmPRnZManQwWEdTZjg&export=download).

The data consists of more than a million of tweet, each tweet comes with its content / text and a binary sentiment score (positive / negative feeling).
