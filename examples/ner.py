# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from nltk.parse.corenlp import CoreNLPParser
from examples.example_corpus import corpus_ner
from nlpkit.nlp_feature_extraction import NERPreprocessor, NamedEntitiesCounter


if __name__ == "__main__":
    sf_parser = CoreNLPParser(url='http://localhost:9000/', tagtype='ner')

    # NERPreprocessor
    ner_preprocessor = NERPreprocessor(sf_parser)
    X = ner_preprocessor.fit_transform(corpus_ner)

    print(X)

    # NamedEntitiesCounter
    named_entities_counter = NamedEntitiesCounter(sf_parser)
    X_c = named_entities_counter.fit_transform(corpus_ner)

    print(X_c)

    # Example usage in a pipeline
    # pipeline = Pipeline([
    #     ('ner', NERPreprocessor(sf_parser)),
    #     ('vec', TfidfVectorizer()),
    #     ('clf', BernoulliNB())
    # ])
