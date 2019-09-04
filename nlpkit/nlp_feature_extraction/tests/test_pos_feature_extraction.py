# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from collections import Counter

from unittest import mock
import pytest

from nlpkit.nlp_feature_extraction.pos_features import POSExtractor
from nlpkit.nlp_feature_extraction.pos_features import POSTagPreprocessor
import nlpkit.nlp_feature_extraction.tests.test_data.doc_test_data as test_data
import nlpkit.nlp_feature_extraction.tests.test_data.nlp_test_data as nlp_test_data


@pytest.fixture(scope="module")
def mockTagger():
    mock_tagger = mock.Mock()
    mock_tagger.tag.side_effect = tag_side_effect
    return mock_tagger


def tag_side_effect(*args, **kwargs):
    if args[0] == test_data.DOC_1.split(' '):
        return nlp_test_data.POS_TAGS_DOC_1
    elif args[0] == test_data.DOC_2.split(' '):
        return nlp_test_data.POS_TAGS_DOC_2


def test_postag_preprocessor(mockTagger):
    expected_results = [
        'The_DT quick_JJ brown_JJ fox_NN jumped_VBZ over_IN the_DT lazy_JJ dog_NN ._.',
        'My_PRP dog_NN also_RB likes_VBZ eating_JJ sausage_NN ._.'
    ]

    pos_tag_preprocessor = POSTagPreprocessor(mockTagger)
    assert pos_tag_preprocessor.transform(test_data.DOCS) == expected_results


def test_pos_extractor(mockTagger):
    expected_results = [
        Counter({
            'DT': 2,
            'JJ': 3,
            'NN': 2,
            'VBZ': 1,
            'IN': 1,
            '.': 1
        }),
        Counter({
            'PRP': 1,
            'NN': 2,
            'RB': 1,
            'VBZ': 1,
            'JJ': 1,
            '.': 1
        }),
    ]

    pos_extractor = POSExtractor(mockTagger)
    assert pos_extractor.transform(test_data.DOCS) == expected_results
