#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:48:17 2018

@author: evita
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

root = "../sarcasm_data/" #put the data in a parent folder named "sarcasm_data"

def load_data_and_split(root):
    if not os.path.exists(root): 
        os.makedirs(root)
    print root
    df = pd.read_csv(root + "train-balanced-sarcasm.csv")
    print "loading file.. OK"
    train, test = train_test_split(df, test_size=0.1, random_state=1)
    train, validate = train_test_split(df, test_size=0.2, random_state=1)
    print "sizes:", "train:", len(train), ", validate:", len(validate), \
                                                        ", test:", len(test)
    train.to_csv(root + 'sarcasm_train.csv', sep=',')
    test.to_csv(root + 'sarcasm_test.csv', sep=',')
    validate.to_csv(root + 'sarcasm_validate.csv', sep=',')
    print "writing splitted files is COMPLETE"
    
    
def load_file(filename, column_names):
    #column_names = ['label', 'comment', 'parent_comment']
    df = pd.read_csv(root + "train-balanced-sarcasm.csv", names = column_names)
    return df


load_data_and_split(root)