# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from examples.example_corpus import corpus
from nlpkit.nlp_feature_extraction import TextStatsExtractor


if __name__ == "__main__":
    # TextStatsExtractor
    stats_extractor = TextStatsExtractor()
    X = stats_extractor.fit_transform(corpus)

    print(X)

    # Example usage in a pipeline
    # pipeline = Pipeline([
    #     ('stats', TextStatsExtractor()),
    #     ('vec', DictVectorizer()),
    #     ('imp', Imputer(strategy='mean', axis=0)),
    #     ('norm', Normalizer()),
    #     ('clf', SVC(kernel='linear', probability=True))
    # ])