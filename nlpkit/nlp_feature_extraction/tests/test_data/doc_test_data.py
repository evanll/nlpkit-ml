# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

DOC_1 = 'The quick brown fox jumped over the lazy dog.'
DOC_2 = 'My dog also likes eating sausage.'
DOCS = [DOC_1, DOC_2]

# Tfidf Vectorizer mock data
TFIDF_MATRIX_DOCS = [[0.23700504099641823,
                      0.0,
                      0.3331023155662109,
                      0.0,
                      0.3331023155662109,
                      0.23700504099641823,
                      0.0,
                      0.3331023155662109,
                      0.3331023155662109,
                      0.3331023155662109,
                      0.0,
                      0.3331023155662109,
                      0.3331023155662109,
                      0.0,
                      0.3331023155662109],
                     [0.29017020899133733,
                      0.4078241041497786,
                      0.0,
                      0.4078241041497786,
                      0.0,
                      0.29017020899133733,
                      0.4078241041497786,
                      0.0,
                      0.0,
                      0.0,
                      0.4078241041497786,
                      0.0,
                      0.0,
                      0.4078241041497786,
                      0.0]]

# The indexes of the words in the vocab list correspond to those in the
# Tfidf matrix
TFIDF_MATRIX_VOCAB = [
    '.',
    'My',
    'The',
    'also',
    'brown',
    'dog',
    'eating',
    'fox',
    'jumped',
    'lazy',
    'likes',
    'over',
    'quick',
    'sausage',
    'the']
