# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

import os

from gensim.models.keyedvectors import KeyedVectors

from examples.example_corpus import corpus
from nlpkit.nlp_feature_extraction import WordEmbedsDocVectorizer

# Replace with the path of a pre-trained w2v model
WORD_EMBEDDINGS_FILEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'fake-news', 'data', 'resources', 'word-embeddings',
                 'glove.6B', 'glove.6B.300d.word2vec.txt'))


if __name__ == "__main__":
    word2vec = KeyedVectors.load_word2vec_format(
        WORD_EMBEDDINGS_FILEPATH, binary=False)

    w2v_vectorizer = WordEmbedsDocVectorizer(word2vec, tfidf_weights=True)
    X = w2v_vectorizer.fit_transform(corpus)

    print(X)

    # Example usage in a pipeline
    # pipeline = Pipeline(
    #     ('vec', WordEmbedsDocVectorizer(word2vec, tfidf_weights=True)),
    #     ('clf', SVC(kernel='linear', C=1, probability=True))
    # ])
