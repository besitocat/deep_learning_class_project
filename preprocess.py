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
from langdetect import detect

root_sarcasm_data_dir = "../sarcasm_data/" #put the data (train-balanced-sarcasm.csv)
                                        #in a parent folder named "sarcasm_data"
root_sarcasm_models_dir = "../sarcasm_models/"
sarcasm_file = "train-balanced-sarcasm.csv"
train_file = 'nostopwords_train.csv'
test_file = 'nostopwords_test.csv'
validate_file = 'nostopwords_validate.csv'
train_file_cleaned =  "train_cleaned.csv"
validate_file_cleaned = "validate_cleaned.csv"
test_file_cleaned = "test_cleaned.csv"
root_yelp_data_dir = "../yelp_data/" #put the data (review.csv)
                                        #in a parent folder named "yelp_data"
yelp_file = "review_sampled.csv"

def prepare_for_yelp(is_yelp=True):
    if is_yelp:
        global root_sarcasm_data_dir, sarcasm_file
        root_sarcasm_data_dir = root_yelp_data_dir
        sarcasm_file = yelp_file
        print "replaced with new:", root_sarcasm_data_dir, sarcasm_file

def load_file_sarcasm(filename):
    print "\n**** LOADING FILE: " + filename + "..."
    column_names = ['label','comment']#, 'parent_comment'] TODO: maybe merge this as well?
    df = pd.read_csv(root_sarcasm_data_dir+filename, usecols = column_names,sep=",")
    print "initial shape:", df['comment'].shape
    df['comment'].replace('', np.nan, inplace=True)
    df.dropna(subset=['comment'], inplace=True) #remove NaNs
    print "*** removed empty comments ***"
    print "new shape:", df['comment'].shape
    return df

def load_file_sarcasm2(df):
    print "initial shape:", df['comment'].shape
    df['comment'].replace('', np.nan, inplace=True)
    df.dropna(subset=['comment'], inplace=True) #remove NaNs
    print "*** removed empty comments ***"
    print "new shape:", df['comment'].shape
    return df

def load_preprocessed_file2(filename):
    print "\n**** LOADING PREPROCESSED FILE: " + filename + " ..."
    column_names = ['label', 'clean_comments']
    df = pd.read_csv(root_sarcasm_data_dir + filename, usecols=column_names,
                     converters={"clean_comments": literal_eval}, sep=",")
    return df


def load_preprocessed_file(filename):
    print "\n**** LOADING PREPROCESSED FILE: " + filename + " ..."
    column_names = ['label','clean_comments']
    df = pd.read_csv(root_sarcasm_data_dir + filename, usecols = column_names,
                                     converters={"clean_comments": literal_eval}, sep=",")
    df_data = df.drop(['label'], axis=1)
    df_target = df.drop(['clean_comments'], axis=1)
    print "total positive vs negative examples in dataset:\n", df_target['label'].value_counts()
    return df_data, df_target

def load_data(root_sarcasm, subset_size=None):
    print "\n**** Loading data****"
    print "**** loading file.. :" + sarcasm_file
    if not os.path.exists(root_sarcasm):
        os.makedirs(root_sarcasm)
    df = pd.read_csv(root_sarcasm + sarcasm_file,",")
    print("Total loaded data: %d"%len(df))
    if subset_size is not None:
        df=df.sample(n=subset_size)
    return df


def train_val_split_data(df, val_size, random_state=None):
    # here we may need to create multiple splits if we do cross-validation.
    print "\n**** Splitting data into train and validate sets ****"
    train, validate = train_test_split(df, test_size=val_size, random_state=random_state)
    print "train:", train['clean_comments'].head
    train.to_csv(root_sarcasm_data_dir+train_file, sep=',')
    validate.to_csv(root_sarcasm_data_dir+validate_file, sep=',')
    print "\nvalidate:", validate['clean_comments'].head
    print "sizes:", "train:", len(train), ", validate:", len(validate)
    print "**** writing splitted files is COMPLETE ****"
    return train,validate


def train_test_split_data(df, size, random_state=1234):
    print "\n**** Extracting test set. ****"
    train, test = train_test_split(df, test_size=size, random_state=random_state)
    test.to_csv(root_sarcasm_data_dir + test_file, sep=',')
    print "test:"
    print test['clean_comments'].head
    print "sizes:", "test:", len(test)
    #return the remaining dataset after extracting the test set.
    return train, test


def clean_and_split_data(subset_size=50000, remove_stopwords=True, max_seq_length=None, test_size=0.2, val_size=0.1):
    print "\n********** Preparing data for Sarcasm dataset ******************"
    # if running for the first time: you must uncomment all the calls below!!
    print "\n**** Preparing data for preprocessing...."
    # load data
    if os.path.exists(root_sarcasm_data_dir + "subset_"+str(subset_size)+ "_" + sarcasm_file):
        print("Loading %s"%root_sarcasm_data_dir + "subset_"+str(subset_size)+ "_" + sarcasm_file)
        df_sarcasm = pd.read_csv(root_sarcasm_data_dir + "subset_"+str(subset_size)+ "_" + sarcasm_file,sep=",")
    else:
        df_sarcasm = load_data(root_sarcasm_data_dir, subset_size=subset_size)
        if subset_size is not None:
            df_sarcasm.to_csv(root_sarcasm_data_dir + "subset_"+str(subset_size)+ "_" + sarcasm_file, sep=',')
    if not os.path.exists(root_sarcasm_data_dir + test_file):
        # Firstly, extract test data. Always do this with the same random state.
        train = train_test_split_data(df_sarcasm, test_size, random_state=1234)
        # split rest data into train/val sets
        train_val_split_data(train, val_size)

    df_train = load_file_sarcasm(root_sarcasm_data_dir + train_file)
    df_validate = load_file_sarcasm(root_sarcasm_data_dir + validate_file)
    df_test = load_file_sarcasm(test_file)
    print "\n**** Preprocessing data...."

    # preprocess all splits below, if we create multiple splits.
    preprocess_text(df_train, train_file_cleaned, remove_stopwords)
    preprocess_text(df_validate, validate_file_cleaned, remove_stopwords)

    preprocess_text(df_test, test_file_cleaned, remove_stopwords)
    print "**** PREPROCESSING COMPLETED for all files"


def test_with_small_files():
    #function to be used for training with small dataset file (validation file in this case)
    df_validate_data, df_validate_targets = load_preprocessed_file(validate_file_cleaned)
    bag_of_words, vec_model = transform_to_vec_values(df_validate_data, df_validate_targets, "clean_comments", 'validate_with_tfidf.csv')
    y_val = df_validate_targets['label'].values
    return bag_of_words, vec_model, y_val

def remove_stopwords(df, filename):
    print("Removing stopwords")
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(["theres", "would", "could", "ive", "theyre", "dont", "since"])
    df['clean_comments'] = df['clean_comments'].apply(lambda x: [item for item in x \
                                                                 if item not in stopwords])
    df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c == [])
    df.drop(df[df['empty_list_comments'] == True].index, inplace=True)
    print "\ncomments with stopwords removed:" + str(remove_stopwords) + "\n", df['clean_comments'].head()
    print "shape:", df['clean_comments'].shape
    df.to_csv(root_sarcasm_data_dir + filename, sep=',')
    print "**** PREPROCESSING COMPLETED. New file generated: " + filename



def preprocess_text(df, new_filename, remove_stopwords):
    print "\n**** PREPROCESSING STARTED ...."
    print("Removing punctuation")
    df['clean_comments'] = df['comment'].apply(lambda x:''.join([\
          re.sub('[^a-z\s]', '', i.lower()) for i in x if i not in string.punctuation]))
    
    print("Tokenizing")
    df['clean_comments'] = df['clean_comments'].apply(nltk.word_tokenize)
    df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c==[])
    df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    # print "searching for English-only comments"
#    df = search_english_only_comments(df)
    if remove_stopwords:
        print("Removing stopwords")
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
    df.to_csv(root_sarcasm_data_dir + new_filename, sep=',')
    print "**** PREPROCESSING COMPLETED. New file generated: " + new_filename
    
    
def preprocess_text_new(df, truncate_length):
    print "\n**** PREPROCESSING STARTED ...."
    print("Removing punctuation")
    df['clean_comments'] = df['comment'].apply(lambda x:''.join([\
          re.sub('[^a-z\s]', '', i.lower()) for i in x if i not in string.punctuation]))
    
    print("Tokenizing")
    df['clean_comments'] = df['clean_comments'].apply(nltk.word_tokenize)
    df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c==[])
    df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    # print "searching for English-only comments"
#    df = search_english_only_comments(df)
    print "Truncating...."
    df = truncate_document(df, truncate_length)
    df.to_csv(root_sarcasm_data_dir + "sample_review_trunc.csv", sep=',')
    return df

def remove_stopwords_function(df, new_filename):
    print("Removing stopwords")
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(["theres", "would", "could", "ive", "theyre", "dont", "since"])
    df['clean_comments']= df['clean_comments'].apply(lambda x: [item for item in x if item not in stopwords])
    print "shape:", df['clean_comments'].shape
    print "\n",df['clean_comments'].head()
    new_filename = "without_stopwords_" + new_filename
    df.to_csv(root_sarcasm_data_dir + new_filename, sep=',')
    print "New file generated: " + new_filename
    # new_filename = "with_stopwords_" + new_filename
    # df['empty_list_comments'] = df['clean_comments'].apply(lambda c: c==[])
    # df.drop(df[df['empty_list_comments']  == True].index, inplace=True)
    # print "\n",df['clean_comments'].head()
    # print "shape:", df['clean_comments'].shape
    # df.to_csv(root_sarcasm_data_dir + new_filename, sep=',')
    # print "New file generated: " + new_filename
    # print "**** PREPROCESSING COMPLETED."
    
    
def search_english_only_comments(df):
    df['clean_comments'] = df['clean_comments'].apply(lambda x: [item for item in x if detect(item) == 'en'])
    print "done"
    print df.head
    return df

def truncate_document(df, max_length=100, updated_file=None):
    for i,row in df.iterrows():
          cur_length = len(row['clean_comments'])
          if cur_length>max_length:
              df.at[i,'clean_comments'] = row['clean_comments'][:max_length]
    if updated_file:
        df.to_csv(root_sarcasm_data_dir + updated_file, sep=',')
    return df

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


def save_labels(y_train,y_val,y_test,out_folder,suffix=""):
    cPickle.dump(y_train, open(os.path.join(out_folder, "y_train_"+suffix+".pickle"), 'wb'))
    cPickle.dump(y_val, open(os.path.join(out_folder, "y_val_"+suffix+".pickle"), 'wb'))
    cPickle.dump(y_test, open(os.path.join(out_folder, "y_test_"+suffix+".pickle"), 'wb'))
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


def split_before_stopwords(df_sarcasm, test_size=0.2, val_size=0.1):
    # load data
    # print "splitting data into test train validate"
    # if os.path.exists(root_sarcasm_data_dir + filename):
    #     print("Loading %s" % root_sarcasm_data_dir + filename)
    # df_sarcasm = pd.read_csv(root_sarcasm_data_dir + filename,sep=",")
        # Firstly, extract test data. Always do this with the same random state.
    train_df, test = train_test_split_data(df_sarcasm, test_size, random_state=1234)
        # split rest data into train/val sets
#    print train_df['clean_comments'].head
    train, validate = train_val_split_data(train_df, val_size)
    return train, validate,test

def helper(subset_size=10000):
    filename="all_data"
    prepare_for_yelp(is_yelp=True)
    df = load_data(root_sarcasm_data_dir, subset_size=subset_size)
#    if subset_size is not None:
#        df=df.sample(n=subset_size)
#    df.to_csv(root_sarcasm_data_dir + "subset_" + str(subset_size) + "_" + sarcasm_file)
#    df_new = load_file_sarcasm(root_sarcasm_data_dir + "subset_" + str(subset_size) + "_" + sarcasm_file)

    preprocess_text(df, filename, remove_stopwords=False)
    df_new,_ = load_preprocessed_file(
        root_sarcasm_data_dir +"with_stopwords_" + filename)
    df_new_tr = truncate_document(df_new, max_length=300,updated_file=root_sarcasm_data_dir + "max_len_" + str(300) + "_" + filename)

    train = train_test_split_data(df_new_tr, 0.2, random_state=1234)
    # split rest data into train/val sets
    train_val_split_data(train, val_size=0.1)

    df_train,_ = load_preprocessed_file(root_sarcasm_data_dir + train_file)
    df_validate,_ = load_preprocessed_file(root_sarcasm_data_dir + validate_file)
    df_test,_ = load_preprocessed_file(test_file)


    remove_stopwords(df_train, "max_len_" + str(300) + "_" + "train" + "nostopwords")
    remove_stopwords(df_validate, "max_len_" + str(300) + "_" + "val" + "nostopwords")
    remove_stopwords(df_test, "max_len_" + str(300) + "_" + "test" + "nostopwords")
    
    
    
def new_pipeline(subset_size=10000, truncate_length=None):
    prepare_for_yelp(is_yelp=True)
    df = load_data(root_sarcasm_data_dir, subset_size=subset_size)
    df = load_file_sarcasm2(df)
    preprocess_text_new(df, truncate_length)
    print("Loading %s" % root_sarcasm_data_dir+"sample_review_trunc.csv")
    df_trunc=load_preprocessed_file2(root_sarcasm_data_dir+"sample_review_trunc.csv")
    train, validate, test= split_before_stopwords(df_trunc, test_size=0.2, val_size=0.1)
    train =load_preprocessed_file2(root_sarcasm_data_dir+train_file)
    validate =load_preprocessed_file2(root_sarcasm_data_dir+validate_file)
    test = load_preprocessed_file2(root_sarcasm_data_dir+test_file)
    print "GENERATING FINAL FILES...."
    print "test files..."
    remove_stopwords_function(train,test_file)
    print "train files..."
    remove_stopwords_function(validate, train_file)
    print "validate files...."
    remove_stopwords_function(test, validate_file)
    

def main():
    new_pipeline(subset_size=350000, truncate_length=300)
    
if __name__ == '__main__':
    main()
    
