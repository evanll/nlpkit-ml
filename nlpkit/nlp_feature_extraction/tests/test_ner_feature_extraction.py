# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from collections import Counter

from unittest import mock
import pytest

from nlpkit.nlp_feature_extraction.ner_features import NERPreprocessor, NamedEntitiesCounter
import nlpkit.nlp_feature_extraction.tests.test_data.nlp_test_data as nlp_test_data


@pytest.fixture(scope="module")
def mockTagger():
    mock_tagger = mock.Mock()
    mock_tagger.tag.side_effect = tag_side_effect
    return mock_tagger


def tag_side_effect(*args, **kwargs):
    if args[0] == nlp_test_data.NER_DOC_1.split(' '):
        return nlp_test_data.NER_TAGS_DOC_1
    elif args[0] == nlp_test_data.NER_DOC_2.split(' '):
        return nlp_test_data.NER_TAGS_DOC_2


def test_ner_preprocessor(mockTagger):
    expected_results = [
        "_PERSON_ agrees with _PERSON_ `` by voting to give _PERSON_ the benefit of the doubt on _COUNTRY_ . ''",
        'My dog also likes eating sausage .'
    ]

    ner_preprocessor = NERPreprocessor(mockTagger)
    assert ner_preprocessor.transform(
        nlp_test_data.NER_DOCS) == expected_results


def test_named_entities_counter(mockTagger):
    expected_results = [
        Counter({
            'O': 15,
            'PERSON': 3,
            'COUNTRY': 1
        }),
        Counter({'O': 7})
    ]

    named_entities_counter = NamedEntitiesCounter(mockTagger)
    assert named_entities_counter.transform(
        nlp_test_data.NER_DOCS) == expected_results
