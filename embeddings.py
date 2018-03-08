#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:15:17 2018

@author: evita
"""

from gensim.models import KeyedVectors
from tqdm import tqdm
import codecs
import numpy as np

glove_file_english_only_100d = "glove/100d_glove_english_only.txt"

def load_to_gensim(glove_file_english_only):
    model = KeyedVectors.load_word2vec_format(glove_file_english_only_100d, binary=False) #GloVe Model - not updatable
    return model

def get_glove_embed(list_of_words):
    word2vec = load_to_gensim(glove_file_english_only_100d) 
    words = [w for w in list_of_words if w in word2vec.vocab]
    glove_word_dict = dict()
    for w in words:
        glove_word_dict[w] = word2vec[w]
    print "total words not found in Glove:", len(list_of_words)-len(words)
    return glove_word_dict


def get_fasttext_embed(word):
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('wiki.simple.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        fword = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[fword] = coefs
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index[word]


def main():
    list_of_words = ['hello', 'how', 'are', 'oy', 'y444']
    get_glove_embed(list_of_words)
    print get_glove_embed
    ff_embedding= get_fasttext_embed('hello')
    print ff_embedding

if __name__ == "__main__":
    main()