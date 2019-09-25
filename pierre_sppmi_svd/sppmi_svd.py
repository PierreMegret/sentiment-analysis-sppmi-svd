# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import os
import logging
import numpy as np
from math import log
from collections import Counter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

__authors__ = "Pierre Megret"
__emails__ = ['pierre_megret@berkeley.edu']
__version__ = "1.0.0"

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)

# =============================================================================


class SPPMI_SVD(object):
    def __init__(self):
        self.model = {}
        self.filename = {}
        self.size = 0
        self.idx2uni = {}

    def train(self, corpus, min_count, sample, window, context_alpha,
              negative, size, filename, progress_per=100000):

        logging.info("Starting the creation of the model.")

        # Counting the unigram

        logging.info("Starting unigrams count.")

        word_count = Counter()

        for sentence_no, sent in enumerate(corpus):
            if sentence_no % progress_per == 0:
                logging.info("PROGRESS: at sentence #{}, processed {} words, "
                             "keeping {} unique word"
                             .format(sentence_no, len(word_count.values()),
                                     len(word_count.keys())))
            for word in sent:
                word_count[word] += 1

        logging.info("Ending unigrams count.")
        logging.info("collected {} word types from a corpus of {} raw words "
                     "and {} sentences".format(len(word_count.keys()),
                                               len(word_count.values()),
                                               sentence_no))

        logging.info("{} tokens before min_count={}".format(len(word_count),
                     min_count))

        for word, count in word_count.items():
            if count < min_count:
                del count

        logging.info("{} tokens after min_count={}".format(len(word_count),
                     min_count))

        uni2idx = {x: i for i, x in enumerate(word_count.keys())}
        idx2uni = {v: k for k, v in uni2idx.items()}

        # Computing the frequencies of the unigrams
        proba_word = Counter()
        total = sum(word_count.values())

        for k, v in word_count.items():
            proba_word[k] = v / total
        # Computing the rate of downsampling for each unigram
        downsampled = Counter()
        total_downsample = 0
        unique_downsample = 0

        for w, v in proba_word.items():

            word_probability = (np.sqrt(v / sample) + 1) * (sample / v)

            if word_probability < 1.0:
                unique_downsample += 1
                total_downsample += word_count[w] * word_probability
                downsampled[w] = word_probability

            else:
                total_downsample += word_count[w]
                downsampled[w] = 1.0

        logging.info("sample={} downsamples {} most-common words and leaves {}"
                     " untouched".format(sample, unique_downsample,
                                         (len(word_count) -
                                          unique_downsample)))

        logging.info("downsampling leaves estimated {:.0f} word corpus "
                     "({:.2%} of prior {})"
                     .format(total_downsample, (total_downsample /
                             max(sum(word_count.values()), 1)),
                             sum(word_count.values())))

        # Counting co-occurrences
        logging.info("Starting the co-occurence count.")
        cooccur = Counter()

        for n, sent in enumerate(corpus):

            if n % progress_per == 0:
                logging.info("PROGRESS: processing sentence #{}"
                             .format(n))

            sent = [word for word in sent if downsampled[word] == 1 or
                    np.random.random() >= downsampled[word]]
            len_sent = len(sent)

            for idx in range(len_sent):

                if uni2idx.get(sent[idx], -1) == -1:
                    # If word is Out-Of-Vocabulary (OOV), we skip it
                    continue

                for idx2 in range(max(0, idx - window),
                                  min(len_sent, idx + window + 1)):

                    if uni2idx.get(sent[idx2], -1) == -1 or idx2 == idx:
                        # If word is OOV or the target word, we skip it
                        continue

                    cooccur[sent[idx], sent[idx2]] += 1

        logging.info("Ending the co-occurence count.")
        logging.info("{} total bigrams".format(len(cooccur)))

        # Context distribution smoothing &
        # Shifted Positive Pointwise Mutual Information.

        logging.info("Creating the SPPMI matrix")

        sxy = sum(cooccur.values())  # Total count of bigrams occurrence

        proba_context_word = Counter()

        # Context distribution smoothing
        logging.info("Context distribution smoothing with alpha={}"
                     .format(context_alpha))

        total_context = sum([c ** context_alpha for c in word_count.values()])
        for k, v in word_count.items():
            proba_context_word[k] = (v ** context_alpha) / total_context

        # Negative sampling / Shifted PMI
        logging.info("Negative sampling / Shifted PMI with k={}"
                     .format(negative))

        sppmi_samples = Counter()
        data = []
        rows = []
        cols = []

        for (x, y), n in cooccur.items():
            rows.append(uni2idx[x])
            cols.append(uni2idx[y])

            data.append(max(log((n / sxy) / (proba_word[x]) /
                        (proba_context_word[y])) - log(negative), 0))

            sppmi_samples[(x, y)] = data[-1]

        SPPMI = csc_matrix((data, (rows, cols)))

        logging.info("SPPMI matrix created")
        logging.info('{} non-zero elements'.format(SPPMI.count_nonzero()))

        # Single Value Decomposition

        logging.info("Starting the SVD decomposition")
        U, _, _ = svds(SPPMI, k=size, return_singular_vectors='u')

        # Normalization
        norms = np.sqrt(np.sum(np.square(U), axis=1, keepdims=True))
        U /= np.maximum(norms, 1e-7)

        logging.info("Ending the SVD decomposition")
        logging.info("{} word vector of dimension {}"
                     .format(U.shape[0], U.shape[1]))

        # Creating a model
        model = {idx2uni[n]: U[n] for n in range(U.shape[0])}
        logging.info("Model created")

        self.model = model
        self.filename = filename
        self.size = size
        self.idx2uni = idx2uni

        return self.model

    # ==========================

    def save_vectors(self, save=True):
        # Saving a model to a file.
        # One line, one word and its associated vector representation
        string_list = [k + ' ' + ' '.join([repr(val) for val in v])
                       + '\n' for k, v in self.model.items()]

        string = ('{} {}\n'.format(len(self.model), self.size)
                  + ''.join(string_list))

        if save is True:
            with open('{}.vectors'.format(self.filename), 'w',
                      encoding='utf-8') as fp:
                fp.write(string)
            fp.close()

            logging.info("Saved {} word vector under {}"
                         .format(len(string.split("\n")[2:]),
                                 self.filename + '.vectors'))
        else:
            return string

    # ==========================

    def save_model(self):
        # Saving a model as a .npy file
        np.save(self.filename + '.npy', self.model)
        logging.info("Saved the model under {}".format(self.filename + '.npy'))

    # ==========================

    def most_similar(self, positive, topn=5):
        # Cosine similarity for an unigram against all others.
        sim_matrix = np.dot(np.array(list(self.model.values())),
                            self.model[positive])

        list_similar = []

        for i in np.argpartition(-1 * sim_matrix, topn + 1)[:topn + 1]:

            if self.idx2uni[i] == positive:
                continue

            list_similar.append((float(sim_matrix[i]), self.idx2uni[i]))

        return sorted(list_similar, reverse=True)

    # ==========================

    def similarity(self, word1, word2):
        # Computing the similarity between 2 unigrams
        return np.dot(self.model[word1], self.model[word2])
