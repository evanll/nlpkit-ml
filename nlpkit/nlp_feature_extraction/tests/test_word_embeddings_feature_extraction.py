# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

import math

from unittest import mock
import pytest

import numpy as np

from nlpkit.nlp_feature_extraction.word_embeddings_features import WordEmbedsDocVectorizer
import nlpkit.nlp_feature_extraction.tests.test_data.doc_test_data as test_data
import nlpkit.nlp_feature_extraction.tests.test_data.word2vec_test_data as w2v_test_data


@pytest.fixture(scope="module")
def mockWord2vec():
    mock_word2vec = mock.Mock()
    mock_word2vec.vocab = w2v_test_data.DOC
    mock_word2vec.vector_size = w2v_test_data.DIMS
    mock_word2vec.get_vector.side_effect = get_vector_side_effect
    return mock_word2vec


def get_vector_side_effect(*args, **kwargs):
    return w2v_test_data.WORD_VECTORS[args[0]]


def areVectorsEqual(vec_a, vec_b, dims, rel_tol=1e-06):
    """
    Helper function to compare vectors across dimensions with a relative error tolerance

    """
    for i in range(0, dims):
        if not math.isclose(vec_a[i], vec_b[i], rel_tol=rel_tol):
            return False

    return True


class TestString2Vec:
    def test_convert_string2vec_no_weights(self, mockWord2vec):
        wed_vectorizer = WordEmbedsDocVectorizer(mockWord2vec)
        string2vec = wed_vectorizer._convert_string2vec(test_data.DOC_1)
        assert string2vec == w2v_test_data.EXPECTED_WORD2VEC_DOC_REPR

    def test_convert_string2vec_weights(self, mockWord2vec):
        wed_vectorizer = WordEmbedsDocVectorizer(mockWord2vec)
        string2vec = wed_vectorizer._convert_string2vec(
            test_data.DOC_1,
            w2v_test_data.TFIDF_WEIGHTS)
        assert areVectorsEqual(
            string2vec,
            w2v_test_data.EXPECTED_WEIGHTED_WORD2VEC_DOC_REPR,
            w2v_test_data.DIMS)


class TestGetDocTfidfScores:
    def test_get_doc_tfidf_scores(self):
        np_tf_idf_matrix = np.array(test_data.TFIDF_MATRIX_DOCS)
        scores_dict_doc_1 = WordEmbedsDocVectorizer._get_doc_tfidf_scores(
            np_tf_idf_matrix, 0, test_data.TFIDF_MATRIX_VOCAB)

        assert scores_dict_doc_1.get(
            'dog') == test_data.TFIDF_MATRIX_DOCS[0][test_data.TFIDF_MATRIX_VOCAB.index('dog')]
        assert scores_dict_doc_1.get(
            'fox') == test_data.TFIDF_MATRIX_DOCS[0][test_data.TFIDF_MATRIX_VOCAB.index('fox')]

        scores_dict_doc_2 = WordEmbedsDocVectorizer._get_doc_tfidf_scores(
            np_tf_idf_matrix, 1, test_data.TFIDF_MATRIX_VOCAB)
        assert scores_dict_doc_2.get(
            'sausage') == test_data.TFIDF_MATRIX_DOCS[1][test_data.TFIDF_MATRIX_VOCAB.index('sausage')]


class TestWordEmbedsDocVectorizer:
    def test_WordEmbedsDocVectorizer_no_weights(self, mockWord2vec):
        wed_vectorizer = WordEmbedsDocVectorizer(mockWord2vec)
        doc_embeds_df = wed_vectorizer.transform(test_data.DOCS)

        # Test number of docs
        assert len(doc_embeds_df) == 2

        # Test number of dimensions of the resulting doc vectors
        assert len(doc_embeds_df[0]) == w2v_test_data.DIMS

        # Test the doc embedding of the first doc
        assert doc_embeds_df[0] == w2v_test_data.EXPECTED_WORD2VEC_DOC_REPR
