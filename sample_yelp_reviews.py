#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:20:33 2018

@author: evita
"""
import os
import pandas as pd

root_yelp_data_dir = "../yelp_data/" #put the data (review.csv)
                                        #in a parent folder named "yelp_data"

yelp_original_file = "review.csv"
yelp_new_sampled = "review_sampled.csv"

def load_data(root_yelp, yelp_file, subset_size=None):
    print "\n**** Loading data****"
    print "**** loading file.. :" + yelp_file
    if not os.path.exists(root_yelp):
        os.makedirs(root_yelp)
    df = pd.read_csv(root_yelp + yelp_file, sep='\t')
    df = df[df.stars != 6.0]
    df = df[df.stars != 7.0]
    df = df[df.stars != 8.0]
    df = df[df.stars != 20.0]
    df = df[df.stars != 11.0]
    df = df[df.stars != 0.0]
    df = df[df.stars != 10.0]
    df = df[df.stars != 9.0]
    print "initial stats:"
    print df['stars'].value_counts()
    if subset_size is not None:
        df=df.sample(n=subset_size, random_state=123)
    return df

def sample(subset_size = None):
    yelp_df = load_data(root_yelp_data_dir, yelp_original_file, subset_size = subset_size)
    yelp_df.dropna(axis=0, how='any')
    df_new = balance_sample(yelp_df)
    print "after balancing out stats..."
    print df_new['stars'].value_counts()
    df_new.rename(columns={'stars': 'label', 'text': 'comment'}, inplace=True)
    df_new.to_csv(root_yelp_data_dir + yelp_new_sampled, sep=',')
#    print df_new.describe
    
    
def balance_sample(df):
    balanced_df = df.groupby('stars')
    balanced_df = balanced_df.apply(lambda x: x.sample(balanced_df.size().min(), random_state=123)).reset_index(drop=True)
    return balanced_df


def main():
    sample(subset_size=1500000)

if __name__ == '__main__':
    main()