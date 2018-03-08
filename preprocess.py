#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:48:17 2018

@author: evita
"""
import os
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import nltk
import string
import re
import cPickle
import numpy as np
from ast import literal_eval

from features_methods import transform_to_vec_values

root_sarcasm_data_dir = "../sarcasm_data/" #put the data (train-balanced-sarcasm.csv)
                                        #in a parent folder named "sarcasm_data"
root_sarcasm_models_dir = "../sarcasm_models/"
sacrcasm_file = "train-balanced-sarcasm.csv"
train_file = 'sarcasm_train.csv'
test_file = 'sarcasm_test.csv'
validate_file = 'sarcasm_validate.csv'
train_file_cleaned =  "train_cleaned.csv"
validate_file_cleaned = "validate_cleaned.csv"
test_file_cleaned = "test_cleaned.csv"


def load_file_sarcasm(filename):
    print "\n**** LOADING FILE: " + filename + "..."
    column_names = ['label','comment']#, 'parent_comment'] TODO: maybe merge this as well?
    df = pd.read_csv(root_sarcasm_data_dir+filename, usecols = column_names)
    print "initial shape:", df['comment'].shape
    df['comment'].replace('', np.nan, inplace=True)
    df.dropna(subset=['comment'], inplace=True) #remove NaNs
    print "*** removed empty comments ***"
    print "new shape:", df['comment'].shape
    return df

def load_preprocessed_file(filename):
    print "\n**** LOADING PREPROCESSED FILE: " + filename + " ..."
    column_names = ['label','clean_comments']
    df = pd.read_csv(root_sarcasm_data_dir + filename, usecols = column_names,
                                     converters={"clean_comments": literal_eval})
    df_data = df.drop(['label'], axis=1)
    df_target = df.drop(['clean_comments'], axis=1)
    print "total positive vs negative examples in dataset:\n", df_target['label'].value_counts()
    return df_data, df_target

def load_data(root_sarcasm, subset_size=None):
    print "\n**** Loading data****"
    print "**** loading file.. :" + sacrcasm_file
    if not os.path.exists(root_sarcasm):
        os.makedirs(root_sarcasm)
    df = pd.read_csv(root_sarcasm + sacrcasm_file)
    if subset_size is not None:
        df=df.sample(n=subset_size)
    return df


def train_val_split_data(df, val_size, random_state=None):
    # here we may need to create multiple splits if we do cross-validation.
    print "\n**** Splitting data into train and validate sets ****"
    train, validate = train_test_split(df, test_size=val_size, random_state=random_state)
    train.to_csv(root_sarcasm_data_dir+train_file, sep=',')
    validate.to_csv(root_sarcasm_data_dir+validate_file, sep=',')
    print "sizes:", "train:", len(train), ", validate:", len(validate)
    print "**** writing splitted files is COMPLETE ****"


def train_test_split_data(df, size, random_state=1234):
    print "\n**** Extracting test set. ****"
    train, test = train_test_split(df, test_size=size, random_state=random_state)
    test.to_csv(root_sarcasm_data_dir + test_file, sep=',')
    print "sizes:", "test:", len(test)
    #return the remaining dataset after extracting the test set.
    return train


def  clean_and_split_data(subset_size=50000, remove_stopwords=True, test_size=0.2, val_size=0.1):
    print "\n********** Preparing data for Sarcasm dataset ******************"
    # if running for the first time: you must uncomment all the calls below!!
    print "\n**** Preparing data for preprocessing...."
    # load data
    df = load_data(root_sarcasm_data_dir + root_sarcasm_data_dir, subset_size=subset_size)
    # Firstly, extract test data. Always do this with the same random state.
    train = train_test_split_data(df, test_size, random_state=1234)
    # split rest data into train/val sets
    train_val_split_data(train, val_size)

    df_train = load_file_sarcasm(root_sarcasm_data_dir + train_file)
    df_validate = load_file_sarcasm(root_sarcasm_data_dir + validate_file)
    df_test = load_file_sarcasm(test_file)
    print "\n**** Preprocessing data...."
    preprocess_text(df_test, test_file_cleaned, remove_stopwords)

    # preprocess all splits below, if we create multiple splits.
    preprocess_text(df_train, train_file_cleaned, remove_stopwords)
    preprocess_text(df_validate, validate_file_cleaned, remove_stopwords)
    print "**** PREPROCESSING COMPLETED for all files"


def test_with_small_files():
    #function to be used for training with small dataset file (validation file in this case)
    df_validate_data, df_validate_targets = load_preprocessed_file(validate_file_cleaned)
    bag_of_words, vec_model = transform_to_vec_values(df_validate_data, df_validate_targets, "clean_comments", 'validate_with_tfidf.csv')
    y_val = df_validate_targets['label'].values
    return bag_of_words, vec_model, y_val


def preprocess_text(df, new_filename, remove_stopwords):
    print "\n**** PREPROCESSING STARTED ...."
    df['clean_comments'] = df['comment'].apply(lambda x:''.join([\
          re.sub('[^a-z\s]', '', i.lower()) for i in x if i not in string.punctuation]))
    df['clean_comments'] = df['clean_comments'].apply(nltk.word_tokenize)
    df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c==[])
    df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    if remove_stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(["theres", "would", "could", "ive", "theyre", "dont", "since"])
        df['clean_comments']= df['clean_comments'].apply(lambda x: [item for item in x\
                                                          if item not in stopwords])
    else:
        new_filename = "with_stopwords_" + new_filename
    df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c==[])
    df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    print "\ncomments with stopwords removed:"+str(remove_stopwords)+"\n",df['clean_comments'].head()
    print "shape:", df['clean_comments'].shape
#    stemmer = PorterStemmer()
#    df['stemmed_token_comments'] = df['clean_comments'].apply(lambda x: \
#                                                      [stemmer.stem(item) for item in x])
#    print "comments",df['stemmed_token_comments'].head()
    df.to_csv(root_sarcasm_data_dir + new_filename)
    print "**** PREPROCESSING COMPLETED. New file generated: " + new_filename


def get_vocab_filename(remove_stopwords, vocabulary_size):
    if not remove_stopwords:
        bow_vocab_file = root_sarcasm_data_dir + "bow_vocab_" + str(vocabulary_size) + "with_stopwords.pkl"
    else:
        bow_vocab_file = root_sarcasm_data_dir+"bow_vocab_"+ str(vocabulary_size)+".pkl"
    return bow_vocab_file


def get_file_names(remove_stopwords,vocabulary_size):
    if not remove_stopwords:
        global train_file_cleaned, validate_file_cleaned, test_file_cleaned
        train_file_cleaned = "with_stopwords_" + train_file_cleaned
        validate_file_cleaned = "with_stopwords_" + validate_file_cleaned
        test_file_cleaned = "with_stopwords_" + test_file_cleaned
        tf_file = root_sarcasm_data_dir+"bag_of_words_values_"+ str(vocabulary_size)+"with_stopwords.pkl"
        bow_vocab_file = get_vocab_filename(remove_stopwords, vocabulary_size)
    else:
        train_file_cleaned = train_file_cleaned
        validate_file_cleaned = validate_file_cleaned
        test_file_cleaned = test_file_cleaned
        tf_file = root_sarcasm_data_dir+"bag_of_words_values_"+str(vocabulary_size)+".pkl"
        bow_vocab_file = get_vocab_filename(remove_stopwords, vocabulary_size)
    return train_file_cleaned,validate_file_cleaned,test_file_cleaned,tf_file,bow_vocab_file


def save_labels(y_train,y_val,y_test,out_folder):
    cPickle.dump(y_train, open(os.path.join(out_folder, "y_train.pickle"), 'wb'))
    cPickle.dump(y_val, open(os.path.join(out_folder, "y_val.pickle"), 'wb'))
    cPickle.dump(y_test, open(os.path.join(out_folder, "y_test.pickle"), 'wb'))
    print("Labels shapes: ")
    print("y_train shapes: ", y_train.shape)
    print("y_val shapes: " , y_val.shape)
    print("y_test shapes: " , y_test.shape)
    print("\n")

def load_to_gensim(filepath):
    model = KeyedVectors.load_word2vec_format(filepath, binary=False) #GloVe Model - not updatable
    return model


def save_features(x_train,x_val,x_test,out_folder,suffix):
    cPickle.dump(x_train, open(os.path.join(out_folder, "x_train_"+suffix+".pickle"), 'wb'))
    cPickle.dump(x_val, open(os.path.join(out_folder, "x_val_"+suffix+".pickle"), 'wb'))
    cPickle.dump(x_test, open(os.path.join(out_folder, "x_test_"+suffix+".pickle"), 'wb'))
    print("%s features shapes: "%suffix)
    print("x_train shapes: ",  x_train.shape)
    print("x_val shapes: " , x_val.shape)
    print("x_test shapes: " , x_test.shape)
    print("\n")