# SPPMI - SVD : a Python Implementation

## Overview

In the 2014 research paper by Omer Levy and Yoav Goldberg *Neural Word Embedding as Implicit Matrix Factorization* [1](#References), in the 2016 blog post by Sebastian Ruder *On word embeddings - Part 3: The secret ingredients of word2vec* [2](#References), and in the 2017 blog post of Chris Moody *Stop Using Word2Vec* [3](#References), an alternative to Word2Vec is outlined in order to get vector representation of words. Instead of using a shallow neural network and a cost function optimization, this approach only uses word co-occurences, information theory and matrix factorization.

In a nutshell, PMI - SVD can be summed up in 5 steps:
1. Collecting the frequency of each word in the corpus
2. Calculating the probability for word1 to been seen next to word2 (co-occurence matrix)
3. Calculating the Pointwise Mutual Information (PMI) of word1 and word2. In the information theory field, this represents how often word1 and word2 are associated together
4. Reducing the dimension of the PMI matrix with a Singular Value Decomposition.

I added some steps in my implementation of the method, namely:
1. **Minimum Count**: ignoring words below a certain frequency, if a word only appears once or twice in more than 1 million tweets, we won't be able to learn a useful embedding for it.
2. **Downsampling**: just like for Word2Vec, this step reduces the influence of higher-frequency words, such as "the", "she", "have", etc. 
3. **Shifted Positive PMI (SPPMI)**: offsetting the PMI of two words by a constant, this is equivalent to Word2Vec negative sampling. (Referenced both in Omer Levy and Yoav Goldberg research paper [4](#References) and in Sebastian Ruder's [blog post](http://ruder.io/secret-word2vec/index.html#shiftedpmi))

## Acknowledgements

* My implementation of SPPMI - SVD is very simple, if it were to be optimized (Numpy vectorization, multiprocessing or Cython code), it should produce word embeddings much faster than Word2Vec.
* Freely inspired from Alex Klibisz implementation of SPPMI - SVD [*Simple Word Vectors with co-occurrence PMI and SVD*](https://www.kaggle.com/alexklibisz/simple-word-vectors-with-co-occurrence-pmi-and-svd)
* Also inspired by part of `cooccur_matrix.pyx` script from the [dutchembeddings](https://github.com/clips/dutchembeddings) GitHub repository published alongside the research paper [*Evaluating Unsupervised Dutch Word Embeddings as a Linguistic Resource*](http://www.lrec-conf.org/proceedings/lrec2016/pdf/1026_Paper.pdf) [5](#References)
* Another helpful ressource was Chris McCormick [blog post](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

## References

* [1].  Omer Levy and Yoav Goldberg [*Neural Word Embedding as Implicit Matrix Factorization*](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)
* [2]. Sebastian Ruder [*On word embeddings - Part 3: The secret ingredients of word2vec*](http://ruder.io/secret-word2vec/)
* [3]. Chris Moody [*Stop Using Word2Vec*](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/)
* [4]. Mikolov et al. [*Distributed Representations of Words and Phrases and their Compositionality*](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
* [5]. [Repository](https://github.com/clips/dutchembeddings) for the word embeddings experiments described in *Evaluating Unsupervised Dutch Word Embeddings as a Linguistic Resource*, presented at LREC 2016.
```bibtex
@InProceedings{tulkens2016evaluating,
  author = {Stephan Tulkens and Chris Emmery and Walter Daelemans},
  title = {Evaluating Unsupervised Dutch Word Embeddings as a Linguistic Resource},
  booktitle = {Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016)},
  year = {2016},
  month = {may},
  date = {23-28},
  location = {Portorož, Slovenia},
  editor = {Nicoletta Calzolari (Conference Chair) and Khalid Choukri and Thierry Declerck and Marko Grobelnik and Bente Maegaard and Joseph Mariani and Asuncion Moreno and Jan Odijk and Stelios Piperidis},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  isbn = {978-2-9517408-9-1},
  language = {english}
 }
 ```
