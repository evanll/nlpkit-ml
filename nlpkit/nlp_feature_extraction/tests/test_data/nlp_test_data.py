# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

from nltk import Tree

# POS test data
# Mock output from Stanford POS Tagger
POS_TAGS_DOC_1 = [
    ['The', 'DT'],
    ['quick', 'JJ'],
    ['brown', 'JJ'],
    ['fox', 'NN'],
    ['jumped', 'VBZ'],
    ['over', 'IN'],
    ['the', 'DT'],
    ['lazy', 'JJ'],
    ['dog', 'NN'],
    ['.', '.']
]

POS_TAGS_DOC_2 = [
    ['My', 'PRP'],
    ['dog', 'NN'],
    ['also', 'RB'],
    ['likes', 'VBZ'],
    ['eating', 'JJ'],
    ['sausage', 'NN'],
    ['.', '.']
]

# Syntax test data
# Parse tree from Stanford Parser
PARSE_TREE_DOC_1 = "(ROOT\n  (S\n    (NP (DT The) (JJ quick) (JJ brown) (NN fox))\n    (VP (VBZ jumps)\n      (PP (IN over)\n        (NP (DT the) (JJ lazy) (NN dog))))\n    (. .)))"
PARSE_TREE_DOC_2 = "(ROOT\n  (S\n    (NP (PRP$ My) (NN dog))\n    (ADVP (RB also))\n    (VP (VBZ likes)\n      (NP (JJ eating) (NN sausage)))\n    (. .)))"

# Parse tree from Stanford Parser using the Tree structure from NLTK
PARSE_TREE_NLTK_DOC_1 = [
    Tree(
        'ROOT', [
            Tree(
                'S', [
                    Tree(
                        'NP', [
                            Tree(
                                'DT', ['The']), Tree(
                                    'JJ', ['quick']), Tree(
                                        'JJ', ['brown']), Tree(
                                            'NN', ['fox'])]), Tree(
                                                'VP', [
                                                    Tree(
                                                        'VBD', ['jumped']), Tree(
                                                            'PP', [
                                                                Tree(
                                                                    'IN', ['over']), Tree(
                                                                        'NP', [
                                                                            Tree(
                                                                                'DT', ['the']), Tree(
                                                                                    'JJ', ['lazy']), Tree(
                                                                                        'NN', ['dog'])])])]), Tree(
                                                                                            '.', ['.'])])])]
PARSE_TREE_NLTK_DOC_2 = [
    Tree(
        'ROOT', [
            Tree(
                'S', [
                    Tree(
                        'NP', [
                            Tree(
                                'PRP$', ['My']), Tree(
                                    'NN', ['dog'])]), Tree(
                                        'ADVP', [
                                            Tree(
                                                'RB', ['also'])]), Tree(
                                                    'VP', [
                                                        Tree(
                                                            'VBZ', ['likes']), Tree(
                                                                'NP', [
                                                                    Tree(
                                                                        'JJ', ['eating']), Tree(
                                                                            'NN', ['sausage'])])]), Tree(
                                                                                '.', ['.'])])])]

# NER test data
NER_DOC_1 = "Hillary Clinton agrees with John McCain \"by voting to give George Bush the benefit of the doubt on Iran.\""
NER_DOC_2 = "My dog also likes eating sausage."

NER_DOCS = [
    NER_DOC_1,
    NER_DOC_2
]

# Ner tags from Stanford Parser through NLTK
NER_TAGS_DOC_1 = [
    ('Hillary',
     'PERSON'),
    ('Clinton',
     'PERSON'),
    ('agrees',
     'O'),
    ('with',
     'O'),
    ('John',
     'PERSON'),
    ('McCain',
     'PERSON'),
    ('``',
     'O'),
    ('by',
     'O'),
    ('voting',
     'O'),
    ('to',
     'O'),
    ('give',
     'O'),
    ('George',
     'PERSON'),
    ('Bush',
     'PERSON'),
    ('the',
     'O'),
    ('benefit',
     'O'),
    ('of',
     'O'),
    ('the',
     'O'),
    ('doubt',
     'O'),
    ('on',
     'O'),
    ('Iran',
     'COUNTRY'),
    ('.',
     'O'),
    ("''",
     'O')]

NER_TAGS_DOC_2 = [('My', 'O'), ('dog', 'O'), ('also', 'O'),
                  ('likes', 'O'), ('eating', 'O'), ('sausage', 'O'), ('.', 'O')]
