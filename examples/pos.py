# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from nltk.parse.corenlp import CoreNLPParser

from examples.example_corpus import corpus

from nlpkit.nlp_feature_extraction import POSExtractor
from nlpkit.nlp_feature_extraction import POSTagPreprocessor


if __name__ == "__main__":
    sf_parser = CoreNLPParser(url='http://localhost:9000/', tagtype='pos')

    # POSExtractor
    pos_extractor = POSExtractor(sf_parser)
    X = pos_extractor.fit_transform(corpus)

    print(X)

    # POSTagPreprocessor
    pos_preprocessor = POSTagPreprocessor(sf_parser)
    X_pre = pos_preprocessor.fit_transform(corpus)

    print(X_pre)

    # Example usage in a pipeline
    # pipeline = Pipeline([
    #     ('pos', POSExtractor(tagger=sf_parser)),
    #     ('dict_vec', DictVectorizer()),
    #     ('norm', Normalizer()),
    #     ('clf', SVC(kernel='linear', C=1, probability=True))
    # ])