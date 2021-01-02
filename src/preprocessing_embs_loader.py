import gensim
import os
import numpy as np


def load_fasttext_vectors(fname):
    """
    Given a path it loads the FastText pre-trained word embeddings matrix downloaded from the web.
    The matrix is a dictionary data structure, e.g. vacab['dog'] returns the embedding of the word 'dog'.
    :param fname: String - the path of the FastText pre-trained word embeddings matrix
    :return: Dictionary {String:Array} - the dictionary containing for each word its embedding vector
    """
    data = {}
    with open(fname, 'r') as fin:
        for line in fin:
            tokens = line.rstrip().split()
            embedding = np.array([float(val) for val in tokens[1:]])
            data[tokens[0]] = embedding
    return data


def load_google_vectors(fname):
    """
    Given a path it loads the Google pre-trained word embeddings matrix downloaded from the web.
    The matrix is a dictionary data structure, e.g. vacab['dog'] returns the embedding of the word 'dog'.
    :param fname: String - the path of the FastText pre-trained word embeddings matrix
    :return: Dictionary {String:Array} - the dictionary containing for each word its embedding vector
    """
    return gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)


def load_glove_vectors(dname):
    """
    Given a path it loads the FastText pre-trained word embeddings matrix downloaded from the web.
    The matrix is a dictionary data structure, e.g. vacab['dog'] returns the embedding of the word 'dog'.
    :param fname: String - the path of the FastText pre-trained word embeddings matrix
    :return: Dictionary {String:Array} - the dictionary containing for each word its embedding vector
    """
    data = {}
    for filename in os.listdir(dname):
        with open(dname + "/" + filename, 'r') as fin:
            for line in fin:
                tokens = line.rstrip().split()
                embedding = np.array([float(val) for val in tokens[1:]])
                data[tokens[0]] = embedding
    return data
