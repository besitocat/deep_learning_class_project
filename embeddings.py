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
from gensim.models.wrappers import FastText


glove_file_english_only_100d = "glove/100d_glove_english_only.txt"
fast_text_file = "../fasttext/wiki.simple" #the binary version of FFtext


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


def get_fasttext_embed_for_word(word):
    ff_model = FastText.load_fasttext_format(fast_text_file)
    return ff_model[word]


def get_fasttext_embed(list_of_words):
    words_embed_dict = dict()
    ff_model = FastText.load_fasttext_format(fast_text_file)
    for w in list_of_words:
        words_embed_dict[w] = ff_model[w]
    return words_embed_dict


def main():
    list_of_words = ['hello', 'how', 'are', 'oy', 'y444']
    glove_embed = get_glove_embed(list_of_words)
    fftext_embed = get_fasttext_embed(list_of_words)
#    print glove_embed
#    print get_fasttext_embed_for_word('gegaegag')
#    print fftext_embed
    
if __name__ == "__main__":
    main()