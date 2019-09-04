# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from unittest import mock
import pytest

from nlpkit.nlp_feature_extraction.syntax_features import CFGExtractor
import nlpkit.nlp_feature_extraction.tests.test_data.doc_test_data as test_data
import nlpkit.nlp_feature_extraction.tests.test_data.nlp_test_data as nlp_test_data


@pytest.fixture(scope="module")
def mockParser():
    mock_parser = mock.Mock()
    mock_parser.raw_parse.side_effect = tag_side_effect
    return mock_parser


def tag_side_effect(*args, **kwargs):
    if args[0] == test_data.DOC_1:
        return iter(nlp_test_data.PARSE_TREE_NLTK_DOC_1)
    elif args[0] == test_data.DOC_2:
        return iter(nlp_test_data.PARSE_TREE_NLTK_DOC_2)


def test_cfg_extractor_nonlexical(mockParser):
    expected_results = [['ROOT -> S',
                         'S -> NP VP .',
                         'NP -> DT JJ JJ NN',
                         'VP -> VBD PP',
                         'PP -> IN NP',
                         'NP -> DT JJ NN'],
                        ['ROOT -> S',
                         'S -> NP ADVP VP .',
                         'NP -> PRP$ NN',
                         'ADVP -> RB',
                         'VP -> VBZ NP',
                         'NP -> JJ NN']]

    cfg_extractor = CFGExtractor(mockParser, include_lexical=False)
    assert cfg_extractor.transform(test_data.DOCS) == expected_results


def test_cfg_extractor_lexical(mockParser):
    expected_results = [['ROOT -> S',
                         'S -> NP VP .',
                         'NP -> DT JJ JJ NN',
                         "DT -> 'The'",
                         "JJ -> 'quick'",
                         "JJ -> 'brown'",
                         "NN -> 'fox'",
                         'VP -> VBD PP',
                         "VBD -> 'jumped'",
                         'PP -> IN NP',
                         "IN -> 'over'",
                         'NP -> DT JJ NN',
                         "DT -> 'the'",
                         "JJ -> 'lazy'",
                         "NN -> 'dog'",
                         ". -> '.'"],
                        ['ROOT -> S',
                         'S -> NP ADVP VP .',
                         'NP -> PRP$ NN',
                         "PRP$ -> 'My'",
                         "NN -> 'dog'",
                         'ADVP -> RB',
                         "RB -> 'also'",
                         'VP -> VBZ NP',
                         "VBZ -> 'likes'",
                         'NP -> JJ NN',
                         "JJ -> 'eating'",
                         "NN -> 'sausage'",
                         ". -> '.'"]]

    cfg_extractor = CFGExtractor(mockParser, include_lexical=True)
    assert cfg_extractor.transform(test_data.DOCS) == expected_results
