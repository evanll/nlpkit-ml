[![Build Status](https://travis-ci.org/evanll/nlpkit-ml.svg?branch=master)](https://travis-ci.org/evanll/nlpkit-ml)

# NLPkit - Transformers for text classification
A library of scikit compatible text transformers, that are ready to be integrated in an NLP pipeline for various classification tasks.  

## Project structure
    .
    ├── nlpkit
    │   ├── __init__.py
    │   └── nlp_feature_extraction
    │       ├── __init__.py
    │       ├── liwc
    │       ├── word_embeddings_features.py
    │       ├── syntax_features.py
    │       ├── ner_features.py
    │       ├── pos_features.py
    │       ├── liwc_features.py
    │       ├── text_statistics_features.py
    │       └── tests
    │           ├── __init__.py
    │           ├── test_data
    │           ├── test_liwc_feature_extraction.py
    │           ├── test_ner_feature_extraction.py
    │           ├── test_pos_feature_extraction.py
    │           ├── test_syntax_feature_extraction.py
    │           └── test_word_embeddings_feature_extraction.py
    ├── examples
    ├── README.md
    ├── LICENCE
    ├── requirements.txt
    └── setup.py

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

1. Python 3.6
2. Stanford CoreNLP Server for some transformers
3. Pre-trained word vectors for the w2v transformer

### Stanford CoreNLP Server with Docker
Stanford CoreNLP is required for constituency parsing, POS and NER tagging.

The easiest way is to have a CoreNLP Server running is to use Docker. You can find a Dockerfile and instructions to have the server running at [Stanford CoreNLP Server - Docker](https://github.com/evanlal/stanford-corenlp-docker).

### Word vectors
If you don't want to train your own word embeddings, you can download pre-trained word vectors from the [Stanford GloVe project](http://nlp.stanford.edu/projects/glove). 
For example, the Wikipedia model has a vocabulary of 400K words, represented using 300 dimensional vectors. 
The word vectors come in the GloVe format and need to be converted into the word2vec format. 
While the formats are almost identical, you can use gensim to do the conversion.

```
python -m gensim.scripts.glove2word2vec -i glove.txt -o word2vec.txt
```

## List of transformers
- POSTagPreprocessor: Pre-processes text documents by tagging each word in the form of word_TAG_ e.g. what_WP. Can be used to generate POS tagged n-grams
- NERPreprocessor: Pre-processes text documents by replacing named entities with generic tags e.g. PERSON, LOCATION
- WordEmbedsDocVectorizer: Converts text documents to word2vec based document vector representations. It maps 
the words of a document to word2vec vectors, and averages them across dimensions to produce a document vector
representation
- POSExtractor: Extracts Parts of Speech (POS) counts for a collection of text documents
- CFGExtractor: Extracts the Context Free Grammar (CFG) production rules found in a collection of text documents
- NamedEntitiesCounter: Extracts Named Entity counts per entity type (e.g. PERSON) for a collection of text documents
- LIWCExtractor: Extracts proportions of words that fall in the various LIWC categories for a collection of text documents
- TextStatsExtractor: Calculates various text statistics and readability scores for a collection of text documents

## Usage
All the custom transformers extend the BaseEstimator and TransformerMixin and implement the fit and transform methods.
```python
# POSExtractor
sf_parser = CoreNLPParser(url='http://localhost:9000/', tagtype='pos')
pos_extractor = POSExtractor(sf_parser)
X = pos_extractor.fit_transform(corpus)
```


They can also be used in pipelines e.g.

``` python

pipeline = Pipeline([
                ('pre', TextPreprocessor(stemming=False)),
                ('w2v', WordEmbedsDocVectorizer(self._word2vec, tfidf_weights=True)),
                ('clf', SVC(kernel='linear', C=1, probability=True))
           ])
```

For more, you can run the examples included in the examples folder.

## Tests
The Pytest framework is used for unit testing. All of the custom text transformers produced in this project come with an extensive set of unit tests.
To run the tests use:  
`pytest src`

## Project repository
https://github.com/evanll/nlpkit-ml

## Author
Written by Evan Lalopoulos <evan.lalopoulos.2017@my.bristol.ac.uk> as part of his thesis in [Fake News detection using NLP](https://github.com/evanll/fake-news-nlp).

**Evan Lalopoulos** - [evanll](https://github.com/evanll)
