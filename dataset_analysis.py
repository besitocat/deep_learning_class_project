#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:48:22 2018

@author: evita
"""

from __future__ import division
import pandas as pd
import os
import nltk
import string
import re
import seaborn as sns
from collections import Counter
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt

root_sarcasm_data_dir = "../sarcasm_data/" #put the data (train-balanced-sarcasm.csv)

                                        #in a parent folder named "sarcasm_data"
sarcasm_file = "train-balanced-sarcasm.csv"
train_file = 'sarcasm_train.csv'
train_with_stopwords_file = 'with_stopwords_train_cleaned.csv'
test_file = 'sarcasm_test.csv'
validate_file = 'sarcasm_validate.csv'
train_file_cleaned =  "train_cleaned.csv"
validate_file_cleaned = "validate_cleaned.csv"
test_file_cleaned = "test_cleaned.csv"

def load_data(root_sarcasm, sarcasm_file, subset_size=None):
    print "\n**** Loading data****"
    print "**** loading file.. :" + sarcasm_file
    if not os.path.exists(root_sarcasm):
        os.makedirs(root_sarcasm)
    df = pd.read_csv(root_sarcasm + sarcasm_file)
    if subset_size is not None:
        df=df.sample(n=subset_size)
    return df


def plot_lengths(df, column):
    lengths = list()
    words = set()
    threshold_count = 0
    for l in df[column]:
        lengths.append(len(l))
        if len(l)>=50:
            threshold_count +=1
        for w in l:
            words.add(w)
    print "total comments:", len(lengths)
    print "avg comment length:", np.mean(lengths)
    total_words = np.sum(lengths)
    print "total (non unique) words:", total_words
    print "total unique words:", len(words)
    print "total comments above the threshold count (100):", threshold_count
    x = list(range(0,len(lengths)))
#    plt.plot(lengths,x)
    df = pd.DataFrame(lengths, columns=['length'])
    bins = range(0,100,10)
    plt.hist(lengths, bins=bins)
    plt.legend()
    
def load_preprocessed_file(filename):
    print "\n**** LOADING PREPROCESSED FILE: " + filename + " ..."
    column_names = ['label','clean_comments']
    df = pd.read_csv(root_sarcasm_data_dir + filename, usecols = column_names,
                                     converters={"clean_comments": literal_eval})
    df_data = df.drop(['label'], axis=1)
    df_target = df.drop(['clean_comments'], axis=1)
    print "total positive vs negative examples in dataset:\n", df_target['label'].value_counts()
    return df_data, df_target

def get_vocabulary_size(df):
    vocabulary = set()
    total_comments = df.shape[0]
    df2 = df['comment'].astype(str)
    vocabulary_after = set()
#    df2 = df2.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
   # df['clean_comments'] = df['comment'].apply(nltk.word_tokenize)
    df2.str.split().apply(vocabulary.update)
    print "total vocab:", len(vocabulary)
    print "average senquence size:", len(vocabulary)/total_comments
    print "shape1:", df2.shape
    df['clean_comment'] = df['comment'].astype(str).apply(lambda x:''.join([\
          re.sub('[^a-z\s]', '', i.lower()) for i in x if i not in string.punctuation]))
    df['clean_comment'] = df['clean_comment'].apply(nltk.word_tokenize)
    df['empty_list_comments'] = df['clean_comment'].apply(lambda c: c==[])
    df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    print "after shape:", df.shape
    print "df3 after:", df.shape
    df['clean_comment'].apply(vocabulary_after.update)
    print "voc afer:", len(vocabulary_after)
    print "avg seq size:", len(vocabulary_after)/df.shape[0]
#    df4 = df['clean_comments'].apply(lambda x: [item for item in x]).apply(vocabulary_after2.update)
#    total = Counter(" ".join(df['comment']).split(" ")).items()


def dataset_analysis():
    df = load_data(root_sarcasm_data_dir, sarcasm_file)
    print "total comments:", df.shape
    #get_vocabulary_size(df)
    df_train = load_data(root_sarcasm_data_dir, train_file)
    df_train_cleaned_data, df_train_labels = load_preprocessed_file(train_file_cleaned)
    print "results for traun with stopword removal"
#    plot_lengths(df_train_cleaned_data, 'clean_comments')
    print "results without stopword removal"
    df_train_cleaned_with_stop_data, df_train_stop_labels = load_preprocessed_file(train_with_stopwords_file)
    plot_lengths(df_train_cleaned_with_stop_data, 'clean_comments')
    
#    df_val = load_data(root_sarcasm_data_dir, validate_file)
#    df_validate_cleaned_data, df_val_labels = load_preprocessed_file(df_val)
#    print "results for validate with stopword removal"
#    plot_lengths(df_validate_cleaned_data, 'clean_comments')
    
def main():
    dataset_analysis()

if __name__ == '__main__':
    main()