#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:48:17 2018

@author: evita
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
import string
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
import pickle
from sklearn.externals import joblib

root_sarcasm_data_dir = "../sarcasm_data/" #put the data (train-balanced-sarcasm.csv) 
                                        #in a parent folder named "sarcasm_data"
root_sarcasm_models_dir = "../sarcasm_models/"
sacrcasm_file = "train-balanced-sarcasm.csv"
train_file = root_sarcasm_data_dir + 'sarcasm_train.csv'
test_file = root_sarcasm_data_dir + 'sarcasm_test.csv'
validate_file = root_sarcasm_data_dir + 'sarcasm_validate.csv'
train_file_cleaned =  "train_cleaned.csv"
validate_file_cleaned = "validate_cleaned.csv"
test_file_cleaned = "test_cleaned.csv"


def load_data_and_split(root_sarcasm):
    print "\n**** Splitting data into test, train and validate sets ****"
    if not os.path.exists(root_sarcasm): 
        os.makedirs(root_sarcasm)
    df = pd.read_csv(root_sarcasm + sacrcasm_file)
    print "**** loading file.. :" + sacrcasm_file
    train, test = train_test_split(df, test_size=0.1, random_state=1)
    train, validate = train_test_split(train, test_size=0.2, random_state=1)
    print "sizes:", "train:", len(train), ", validate:", len(validate), \
                                                        ", test:", len(test)
    train.to_csv(train_file, sep=',')
    test.to_csv(test_file, sep=',')
    validate.to_csv(validate_file, sep=',')
    print "**** writing splitted files is COMPLETE ****"
    
    
def load_file_sarcasm(filename):
    print "\n**** LOADING FILE: " + filename + "..."
    column_names = ['label','comment']#, 'parent_comment'] TODO: maybe merge this as well?
    df = pd.read_csv(filename, usecols = column_names)
    print "initial shape:", df['comment'].shape
    df['comment'].replace('', np.nan, inplace=True)
    df.dropna(subset=['comment'], inplace=True) #remove NaNs
    print "*** removed empty comments ***"
    print "new shape:", df['comment'].shape
    return df

def load_preprocessed_file(filename):
    print "\n**** LOADING PREPROCESSED FILE: " + filename + " ..."
    column_names = ['label','clean_comments']
    df = pd.read_csv(root_sarcasm_data_dir + filename, usecols = column_names, converters={"clean_comments": literal_eval})
    df_data = df.drop(['label'], axis=1)
    df_target = df.drop(['clean_comments'], axis=1)
    print "total positive vs negative examples in dataset:\n", df_target['label'].value_counts()
    return df_data, df_target

def preprocess_text(df, new_filename):
    print "\n**** PREPROCESSING STARTED ...."
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(["theres", "would", "could", "ive", "theyre", "dont", "since"])
    df['clean_comments'] = df['comment'].apply(lambda x:''.join([\
          re.sub('[^a-z\s]', '', i.lower()) for i in x if i not in string.punctuation]))
    df['clean_comments'] = df['clean_comments'].apply(nltk.word_tokenize)
    df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c==[])
    df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    df['clean_comments']= df['clean_comments'].apply(lambda x: [item for item in x\
                                                          if item not in stopwords])
    df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c==[])
    df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    print "\ncomments without stopwords\n",df['clean_comments'].head()
    print "shape:", df['clean_comments'].shape
#    stemmer = PorterStemmer()
#    df['stemmed_token_comments'] = df['clean_comments'].apply(lambda x: \
#                                                      [stemmer.stem(item) for item in x])
#    print "comments",df['stemmed_token_comments'].head()
    df.to_csv(root_sarcasm_data_dir + new_filename)
    print "**** PREPROCESSING COMPLETED. New file generated: " + new_filename


def prepare_data():
    print "\n**** Preparing data for preprocessing...."
    load_data_and_split(root_sarcasm_data_dir) 
    df_train = load_file_sarcasm(train_file)
    df_validate = load_file_sarcasm(validate_file)
    df_test = load_file_sarcasm(test_file)
    return df_train, df_validate, df_test
    
def preprocess_data(df_train, df_validate, df_test):
    print "\n**** Preprocessing data...."
    preprocess_text(df_train, train_file_cleaned) 
    preprocess_text(df_validate, validate_file_cleaned) 
    preprocess_text(df_test, test_file_cleaned) 
    print "**** PREPROCESSING COMPLETED for all files"

def transform_to_vec_values(df, df_labels, column_name, filename):
    print "\n**** Transforming to numerical values.... to file: ", filename
#    vec_model = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, use_idf=False)
    vec_model = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    bag_of_words = vec_model.fit_transform(df[column_name])
    print "shape of transformation:", bag_of_words.shape
    print "vocabulary size:", len(vec_model.get_feature_names())
    #df_new = pd.DataFrame(bag_of_words.toarray(), columns=v.get_feature_names())
    labels = list(df_labels['label'])
    df.insert(loc=0, column='label', value=labels)
    df['tf-idf-transform'] = list(bag_of_words)
    #df_final = pd.concat([df, df_new], axis=1)
    df.to_csv(root_sarcasm_data_dir + filename)
    print "**** Transforming to TF-IDF is COMPLETE. New column - tf-idf-transform added to file: "\
                                    + filename + " and the vectorized was saved into a pickle file"
    return bag_of_words, vec_model

def load_transformed_to_tfidf(filename, model_filename):
    print "\n**** LOADING TF-IDF TRANSFORMED FILE: " + filename + " ..."
    #column_names = ['tf-idf-transform']
    df = pd.read_csv(root_sarcasm_data_dir + filename)
    df_data = df.drop(['label'], axis=1)
    return df_data


def pipeline_sarcasm():
    print "\n********** Preparing data for Sarcasm dataset ******************"
    #if running for the first time: you must uncomment all the calls below!!
    df_train, df_validate, df_test = prepare_data() # RUN this once and then use the new files generated
    preprocess_data(df_train, df_validate, df_test)
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_files()
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_all_files():
    df_train_data, df_train_targets = load_preprocessed_file(train_file_cleaned)
    df_validate_data, df_validate_targets = load_preprocessed_file(validate_file_cleaned)
    df_test_data, df_test_targets = load_preprocessed_file(test_file_cleaned)
    bag_of_words, vec_model = transform_to_vec_values(df_train_data, df_train_targets, "clean_comments", 'train_with_tfidf.csv')
    X_train = bag_of_words
    X_val = vec_model.transform(df_validate_data["clean_comments"])
    X_test = vec_model.transform(df_test_data["clean_comments"])
    y_val = df_validate_targets['label'].values
    y_train = df_train_targets['label'].values
    y_test = df_test_targets['label'].values
    return X_train, y_train, X_val, y_val, X_test, y_test

def test_with_small_files():
    #function to be used for training with small dataset file (validation file in this case)
    df_validate_data, df_validate_targets = load_preprocessed_file(validate_file_cleaned)
    bag_of_words, vec_model = transform_to_vec_values(df_validate_data, df_validate_targets, "clean_comments", 'validate_with_tfidf.csv')
    y_val = df_validate_targets['label'].values
    return bag_of_words, vec_model, y_val
    
def main():
    #pipeline_sarcasm()
    pass
    
if __name__ == '__main__':
    main()
