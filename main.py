#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:57:40 2018

@author: evita
"""

import data_preprocess_transform as data_prep
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB

def create_model(input_size):
    '''
    Simple NN model. @Efi: do your magic here! 
    '''
    print "\nBuidling the model...."
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_size))
    # Adding the second hidden layer
   # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier 

    
def train_model(classifier, X_train, y_train, epochs):
    print "\nTraining the model...."
    classifier.fit(X_train, y_train, batch_size = 4000, epochs = epochs)
    return classifier

def predict(classifier, X_val):
    print "\nPredicting...."
    y_pred = classifier.predict(X_val)
    y_pred = (y_pred>0.5) #convert to binary output preedictions
    return y_pred

def evaluate(y_true, y_pred):
    print "\nEvaluating..."
    conf_matrix = confusion_matrix(y_true, y_pred)
    print "\nConfusion matrix:\n", conf_matrix
    target_names = ['class 0 (not sarcasm)', 'class 1 (sarcasm)']
    print(classification_report(y_true, y_pred, target_names=target_names))
    

def pipeline(is_first_run=False, train_with_real_data=True, epochs = 50, sample_test_file=None):
    '''
    Pipeline for training and testing the NN model. Please provide train_with_real_data = False, 
    if you want to test it with small files (validation file is used for training and a 
    10 comment file is used for testing/validation)
    '''
    if train_with_real_data:
        if is_first_run:
            X_train, y_train, X_val, y_val, X_test, y_test = data_prep.pipeline_sarcasm() # run this once to get all the data ready
        else:
            X_train, y_train, X_val, y_val, X_test, y_test =  data_prep.load_all_files() # run this if you have already the preprocessed files
    else:
        X_train, vec_model, y_train = data_prep.test_with_small_files()
        df_small_data, y_val = data_prep.load_preprocessed_file(sample_test_file)
        X_val = vec_model.transform(df_small_data["clean_comments"])
    input_size = X_train.shape[1]
    nn_model = create_model(input_size)
    nn_model = train_model(nn_model, X_train, y_train, epochs)
    y_pred = predict(nn_model, X_val)
    evaluate(y_val, y_pred)


def naive_bayes_pipeline(train_with_real_data=True, sample_test_file=None):
    '''
    a test function that the data are in correct format and can be used for 
    training a simple NB model. Please provide train_with_real_data = False, 
    if you want to test it with small files (validation file is used for training 
    and a 10 comment file is used for testing/validation)
    '''
    if train_with_real_data:
        X_train, y_train, X_val, y_val, X_test, y_test = data_prep.pipeline_sarcasm()
    else:
        X_train, vec_model, y_train = data_prep.test_with_small_files()
        df_small_data, y_val = data_prep.load_preprocessed_file(sample_test_file)
        X_val = vec_model.transform(df_small_data["clean_comments"])
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    print "predicting values"
    print "test shape:", X_val.shape
    y_pred = predict(classifier, X_val)
    evaluate(y_val, y_pred)
                   
def main():
    
    pipeline(is_first_run=False, train_with_real_data=False,
             epochs=3, sample_test_file=data_prep.root_sarcasm_data_dir + "small_train.csv")  #training with small files and testing with small file
    
#    naive_bayes_pipeline(train_with_real_data=False, 
#                         sample_test_file=data_prep.root_sarcasm_data_dir + "small_train.csv")   # a test with a simple NB model
    
if __name__ == '__main__':
    main()