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
import cPickle

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
    train.to_csv(root_sarcasm_data_dir+train_file, sep=',')
    test.to_csv(root_sarcasm_data_dir+test_file, sep=',')
    validate.to_csv(root_sarcasm_data_dir+validate_file, sep=',')
    print "**** writing splitted files is COMPLETE ****"
    
    
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


def prepare_data():
    print "\n**** Preparing data for preprocessing...."
    load_data_and_split(root_sarcasm_data_dir+root_sarcasm_data_dir) 
    df_train = load_file_sarcasm(root_sarcasm_data_dir+train_file)
    df_validate = load_file_sarcasm(root_sarcasm_data_dir+validate_file)
    df_test = load_file_sarcasm(test_file)
    return df_train, df_validate, df_test
    
def preprocess_data(df_train, df_validate, df_test, remove_stopwords):
    print "\n**** Preprocessing data...."
    preprocess_text(df_train, train_file_cleaned, remove_stopwords) 
    preprocess_text(df_validate, validate_file_cleaned, remove_stopwords) 
    preprocess_text(df_test, test_file_cleaned, remove_stopwords) 
    print "**** PREPROCESSING COMPLETED for all files"

def transform_to_vec_values(df, df_labels, column_name, filename, vocabulary_size, picklefile1, picklefile2):
    print "\n**** Transforming to numerical values.... to file: ", filename
#    vec_model = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, use_idf=False)
    vec_model = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x,max_features=vocabulary_size)
    bag_of_words = vec_model.fit_transform(df[column_name])
    #save to pickle file
    cPickle.dump(bag_of_words, open(root_sarcasm_data_dir+ picklefile1, 'wb'))#"bag_of_words_values_"+
                                   #str(vocabulary_size)+".pkl", 'wb'))
    print "shape of transformation:", bag_of_words.shape
    cPickle.dump(vec_model.vocabulary_, open(root_sarcasm_data_dir+picklefile2, 'wb'))#"bow_vocab_"+
                                   #str(vocabulary_size)+".pkl", 'wb'))
    print "vocabulary size:", len(vec_model.get_feature_names())
    print "**** Transforming to TF-IDF ****"
    labels = list(df_labels['label'])
    df.insert(loc=0, column='label', value=labels)
    df['tf-idf-transform'] = list(bag_of_words)
    df.to_csv(root_sarcasm_data_dir + filename)
    print "**** Transforming to TF-IDF is COMPLETE. New column - tf-idf-transform added to file: "\
                                    + filename + " and the vectorized was saved into a pickle file"
    return bag_of_words, vec_model


def pipeline_sarcasm(vocabulary_size, load_vocab_file, remove_stopwords):
    print "\n********** Preparing data for Sarcasm dataset ******************"
    #if running for the first time: you must uncomment all the calls below!!
    df_train, df_validate, df_test = prepare_data() # RUN this once and then use the new files generated
    preprocess_data(df_train, df_validate, df_test, remove_stopwords)
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_files(vocabulary_size, load_vocab_file, remove_stopwords)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_all_files(vocabulary_size, load_vocab_file, remove_stopwords):
    if not remove_stopwords:
        global train_file_cleaned, validate_file_cleaned, test_file_cleaned
        train_file_cleaned = "with_stopwords_" + train_file_cleaned
        validate_file_cleaned = "with_stopwords_" + validate_file_cleaned
        test_file_cleaned = "with_stopwords_" + test_file_cleaned
        tf_file = root_sarcasm_data_dir+"bag_of_words_values_"+ str(vocabulary_size)+"with_stopwords.pkl"
        bow_vocab_file = root_sarcasm_data_dir+"bow_vocab_"+ str(vocabulary_size)+"with_stopwords.pkl"
    else:
        tf_file = root_sarcasm_data_dir+"bag_of_words_values_"+str(vocabulary_size)+".pkl"
        bow_vocab_file = root_sarcasm_data_dir+"bow_vocab_"+ str(vocabulary_size)+".pkl"
    df_train_data, df_train_targets = load_preprocessed_file(root_sarcasm_data_dir+train_file_cleaned)
    df_validate_data, df_validate_targets = load_preprocessed_file(root_sarcasm_data_dir+validate_file_cleaned)
    df_test_data, df_test_targets = load_preprocessed_file(root_sarcasm_data_dir+test_file_cleaned)
    if load_vocab_file:
        bag_of_words = cPickle.load(open(tf_file))
        print "bag of words size:", bag_of_words.shape
        vocab = cPickle.load(open(bow_vocab_file))
        vec_model = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, vocabulary=vocab)
    else:
        bag_of_words, vec_model = transform_to_vec_values(df_train_data, df_train_targets,
                            "clean_comments", 'train_with_tfidf.csv',vocabulary_size,tf_file,bow_vocab_file)
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
