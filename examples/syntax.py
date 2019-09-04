# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from nltk.parse.corenlp import CoreNLPParser
from nlpkit.examples.example_corpus import corpus
from nlpkit.nlp_feature_extraction import CFGExtractor


if __name__ == "__main__":
    sf_parser = CoreNLPParser(url='http://localhost:9000/')

    # CFGExtractor
    cfg_extractor = CFGExtractor(sf_parser)
    X = cfg_extractor.fit_transform(corpus)

    print(X)

    # Example usage in a pipeline
    # pipeline = Pipeline([
    #     ('cfg', CFGExtractor(parser=self._sf_parser, include_lexical=True)),
    #     ('vec', TfidfVectorizer(preprocessor=dummy_tokenizer, tokenizer=dummy_tokenizer0)),
    #     ('clf', BernoulliNB())
    # ])
