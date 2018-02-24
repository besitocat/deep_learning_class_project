#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:57:40 2018

@author: evita
"""

import data_preprocess_transform as data_prep
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB

def create_model(input_size):
    print "\ncreating model...."
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = input_size))
    # Adding the second hidden layer
   # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier 
    
def train_model(classifier, X_train, y_train, epochs):
    print "\ntraining model...."
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = epochs)
    return classifier


def predict(classifier, X_val):
    print "\npredicting...."
    y_pred = classifier.predict(X_val)
    y_pred = (y_pred>0.5) #convert to binary output preedictions
    return y_pred

def evaluate(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    print "\nConfusion matrix:\n", conf_matrix
    target_names = ['class 0 (not sarcasm)', 'class 1 (sarcasm)']
    print(classification_report(y_true, y_pred, target_names=target_names))
    

def pipeline():
    X_train, y_train, X_val, y_val, X_test, y_test = data_prep.pipeline_sarcasm() #run this to get all the data ready
    classifier = create_model()
    trained_classifier = predict(classifier, X_train)
    y_pred = predict(trained_classifier, X_val)
    evaluate(y_val, y_pred)
    
def main():
#    pipeline() #run this to train with real data and test on validation data
    '''
    simple NB model to test if data are correctly transformed: with validation file as train and a small file for testing
    '''
    classifier = MultinomialNB()
    X_val, vec_model, y_val = data_prep.test_with_small_files()
    input_size = X_val.shape[1]
    print "Input size:", input_size
    print "training classifier..."
    classifier.fit(X_val, y_val)
    print "getting test files"
    df_small_data, df_small_targets = data_prep.load_preprocessed_file(data_prep.root_sarcasm_data_dir + "small_train.csv")
    test_bow = vec_model.transform(df_small_data["clean_comments"])
    print "predicting values"
    print "test shape:", test_bow.shape
    y_pred = predict(classifier, test_bow)
    evaluate(df_small_targets, y_pred)
    
    
    print "\n\ntesting for Keras"
    nn_model = create_model(input_size)
    nn_model = train_model(nn_model, X_val, y_val, 50)
    y_pred_nn = predict(nn_model, test_bow)
    evaluate(df_small_targets, y_pred_nn)
    
    
if __name__ == '__main__':
    main()