# -*- coding: utf-8 -*-
"""
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk>
University of Bristol, May 2018
Copyright (C) - All Rights Reserved
"""

DOC = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumped",
    "over",
    "the",
    "lazy",
    "dog"
]

# Mock word2vec vectors for words in doc1 (4 dimensions)
DIMS = 4
WORD_VECTORS = {
    "the": [0.15702699683606625, -0.10899316892027855, 0.05305000301450491, 0.08994766200582187],
    "quick": [0.053749833876887955, -0.16255000730355582, -0.24597550307710966, 0.16129433192933598],
    "brown": [-0.08443800065045555, -0.08379416881750028, -0.008435000975926718, -0.8742783293128014],
    "fox": [-0.049120450392365456, -0.3226288314908743, 0.20258000399917364, -0.17307000110546747],
    "jumped": [0.12413849992056687, -0.1456831346343582, 0.12768765880415836, 0.08058449920887749],
    "over": [0.12997816782444715, 0.15702699683606625, 0.10122733470052481, -0.14689783255259195],
    "lazy": [0.05609016430874666, 0.13725333474576473, -0.21280082998176417, 0.45069000124931335],
    "dog": [0.15689266535143057, -0.13407894922420382, 0.13273133585850397, -0.14689783255259195]
}

# Tfidf for words in the doc
TFIDF_WEIGHTS = {
    "the": 0.01,
    "quick": 0.2,
    "brown": 0.2,
    "fox": 0.1,
    "jumped": 0.02,
    "over": 0.01,
    "lazy": 0.05,
    "dog": 0.1
}

WEIGHTS_SUM = sum([TFIDF_WEIGHTS[word] for word in DOC])

# Expected word2vec doc representation (avg per dimension)
# [0.07792720821237674, -0.08582678863657983, 0.022568333928507784, -0.052075537680475806]
EXPECTED_WORD2VEC_DOC_REPR = [0, 0, 0, 0]
for i in range(0, DIMS):
    dim_sum = 0
    for word in DOC:
        dim_sum += WORD_VECTORS[word][i]

    EXPECTED_WORD2VEC_DOC_REPR[i] = dim_sum / len(DOC)


# Expected weighted word2vec doc representation (weighted avg per dimension)
# [0.0205245542428678, -0.1308574323730898, -0.03337711677221315, -0.21445345411609332]
EXPECTED_WEIGHTED_WORD2VEC_DOC_REPR = [0, 0, 0, 0]
for i in range(0, DIMS):
    dim_sum = 0
    for word in DOC:
        dim_sum += WORD_VECTORS[word][i] * TFIDF_WEIGHTS[word]

    EXPECTED_WEIGHTED_WORD2VEC_DOC_REPR[i] = dim_sum / WEIGHTS_SUM
