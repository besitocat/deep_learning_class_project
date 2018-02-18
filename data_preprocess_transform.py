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

root_sarcasm = "../sarcasm_data/" #put the data (train-balanced-sarcasm.csv) 
                                        #in a parent folder named "sarcasm_data"
sacrcasm_file = "train-balanced-sarcasm.csv"
train_file = root_sarcasm + 'sarcasm_train.csv'
test_file = root_sarcasm + 'sarcasm_test.csv'
validate_file = root_sarcasm + 'sarcasm_validate.csv'

def load_data_and_split(root_sarcasm):
    if not os.path.exists(root_sarcasm): 
        os.makedirs(root_sarcasm)
    df = pd.read_csv(root_sarcasm + sacrcasm_file)
    print "**** loading file.. OK ****"
    train, test = train_test_split(df, test_size=0.1, random_state=1)
    train, validate = train_test_split(df, test_size=0.2, random_state=1)
    print "sizes:", "train:", len(train), ", validate:", len(validate), \
                                                        ", test:", len(test)
    train.to_csv(train_file, sep=',')
    test.to_csv(test_file, sep=',')
    validate.to_csv(validate_file, sep=',')
    print "**** writing splitted files is COMPLETE ****"
    
    
def load_file_sarcasm(filename):
    column_names = ['label','comment']#, 'parent_comment'] TODO: maybe merge this as well?
    df = pd.read_csv(filename, usecols = column_names)
    print "initial shape:", df['comment'].shape
    df['comment'].replace('', np.nan, inplace=True)
    df.dropna(subset=['comment'], inplace=True) #remove NaNs
    print "*** removed empty comments ***"
    print "new shape:", df['comment'].shape
    return df

def get_data_target_from_df(df):
    df_data = df.drop(['label'], axis=1)
    df_target = df.drop(['comment'], axis=1)
    return df_data, df_target

def preprocess_text(df, new_filename):
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
    print "\ncomments without stopwords",df['clean_comments'].head()
    print "shape:", df['clean_comments'].shape
#    stemmer = PorterStemmer()
#    df['stemmed_token_comments'] = df['clean_comments'].apply(lambda x: \
#                                                      [stemmer.stem(item) for item in x])
#    print "comments",df['stemmed_token_comments'].head()
    df.to_csv(root_sarcasm + new_filename)
    return df

def prepare_data():
    df_train = load_file_sarcasm(train_file)
    df_validate = load_file_sarcasm(validate_file)
    df_test = load_file_sarcasm(test_file)
    preprocessed_df_train = preprocess_text(df_train,"train_cleaned.csv") 
    preprocessed_df_validate = preprocess_text(df_validate, "validate_cleaned.csv") 
    preprocessed_df_validate = preprocess_text(df_test, "test_cleaned.csv") 
    train_data, train_targets = get_data_target_from_df(preprocessed_df_train)
    validate_data, validate_targets = get_data_target_from_df(preprocessed_df_validate)
    return train_data, train_targets, validate_data, validate_targets
    
def main():
    #load_data_and_split(root_sarcasm) --> RUN this once, then load the new files
    train_data, train_targets, validate_data, validate_targets = prepare_data() # RUN this once and then use the new files generated
    
if __name__ == '__main__':
    main()