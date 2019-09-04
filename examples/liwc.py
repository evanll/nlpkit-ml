# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

import os

from examples.example_corpus import corpus

from nlpkit.nlp_feature_extraction import LIWCExtractor
from nlpkit.nlp_feature_extraction.liwc import Liwc

# Replace with the path of a liwc (.dic) file
LIWC_FILEPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'nlp_feature_extraction', 'liwc', 'resources',
                 'mock_liwc.dic'))

if __name__ == "__main__":
    liwc = Liwc(LIWC_FILEPATH)
    liwc_extractor = LIWCExtractor(liwc)
    X = liwc_extractor.fit_transform(corpus)

    print(X)

    # Example usage in a pipeline
    # pipeline = Pipeline([
    #     ('liwc', LIWCExtractor(liwc_dict)),
    #     ('vec', DictVectorizer()),
    #     ('norm', StandardScaler(with_mean=False)),
    #     ('clf', SVC(kernel='linear', probability=True))
    # ])
