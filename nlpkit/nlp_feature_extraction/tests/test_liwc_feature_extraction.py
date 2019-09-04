# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

import os
from collections import Counter

from nlpkit.nlp_feature_extraction.liwc import Liwc
from nlpkit.nlp_feature_extraction import LIWCExtractor

abs_path = os.path.abspath(os.path.dirname(__file__))
LIWC_PATH = os.path.join(
    abs_path,
    '../liwc/resources/mock_liwc.dic')

DOCS = [
    'I love my dog.',
    'I hate fake news.'
]

liwc = Liwc(LIWC_PATH)


def test_liwc_extractor():
    expected_results = [
        Counter({
            'verb': 0.25,
            'present': 0.25,
            'affect': 0.25,
            'posemo': 0.25,
            'bio': 0.25,
            'sexual': 0.25,
            'social': 0.25,
            'funct': 0.25,
            'pronoun': 0.25,
            'ppron': 0.25,
            'i': 0.25
        }),
        Counter({
            'affect': 0.5,
            'negemo': 0.5,
            'verb': 0.25,
            'present': 0.25,
            'anger': 0.25
        })
    ]

    liwc_extractor = LIWCExtractor(liwc)

    assert liwc_extractor.transform(DOCS) == expected_results


def test_liwc_extractor_with_filters():
    expected_results = [
        Counter({
            'posemo': 0.25
        }), Counter({
            'negemo': 0.5
        })]

    liwc_extractor = LIWCExtractor(liwc, ['posemo', 'negemo'])

    assert liwc_extractor.transform(DOCS) == expected_results
