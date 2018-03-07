import cPickle
import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
import os
import numpy as np
import argparse

def fit_features_noseq(df_train_data, df_train_targets, vocabulary_size, load_vocab_file, tf_file, bow_vocab_file):
    if load_vocab_file:
        bag_of_words = cPickle.load(open(tf_file))
        print "bag of words size:", bag_of_words.shape
        vocab = cPickle.load(open(bow_vocab_file))
        vec_model = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, vocabulary=vocab)
    else:
        bag_of_words, vec_model = preprocess.transform_to_vec_values(df_train_data, df_train_targets,
                            "clean_comments", 'train_with_tfidf.csv',vocabulary_size,tf_file,bow_vocab_file)
    return bag_of_words, vec_model


def transform_features_noseq(vec_model, df_validate_data):
    return vec_model.transform(df_validate_data)


def get_vocab_filename(remove_stopwords, vocabulary_size):
    if not remove_stopwords:
        bow_vocab_file = preprocess.root_sarcasm_data_dir + "bow_vocab_" + str(vocabulary_size) + "with_stopwords.pkl"
    else:
        bow_vocab_file = preprocess.root_sarcasm_data_dir+"bow_vocab_"+ str(vocabulary_size)+".pkl"
    return bow_vocab_file


def get_file_names(remove_stopwords,vocabulary_size):
    if not remove_stopwords:
        global train_file_cleaned, validate_file_cleaned, test_file_cleaned
        train_file_cleaned = "with_stopwords_" + preprocess.train_file_cleaned
        validate_file_cleaned = "with_stopwords_" + preprocess.validate_file_cleaned
        test_file_cleaned = "with_stopwords_" + preprocess.test_file_cleaned
        tf_file = preprocess.root_sarcasm_data_dir+"bag_of_words_values_"+ str(vocabulary_size)+"with_stopwords.pkl"
        bow_vocab_file = get_vocab_filename(remove_stopwords, vocabulary_size)
    else:
        train_file_cleaned = preprocess.train_file_cleaned
        validate_file_cleaned = preprocess.validate_file_cleaned
        test_file_cleaned = preprocess.test_file_cleaned
        tf_file = preprocess.root_sarcasm_data_dir+"bag_of_words_values_"+str(vocabulary_size)+".pkl"
        bow_vocab_file = get_vocab_filename(remove_stopwords, vocabulary_size)
    return train_file_cleaned,validate_file_cleaned,test_file_cleaned,tf_file,bow_vocab_file


def create_noseq_features(vocabulary_size, load_vocab_file, remove_stopwords):
    train_file_cleaned,validate_file_cleaned,test_file_cleaned,tf_file,bow_vocab_file=get_file_names(remove_stopwords,vocabulary_size)
    df_train_data, df_train_targets = preprocess.load_preprocessed_file(preprocess.root_sarcasm_data_dir+train_file_cleaned)
    df_validate_data, df_validate_targets = preprocess.load_preprocessed_file(preprocess.root_sarcasm_data_dir+validate_file_cleaned)
    df_test_data, df_test_targets = preprocess.load_preprocessed_file(preprocess.root_sarcasm_data_dir+test_file_cleaned)

    bag_of_words, vec_model = fit_features_noseq(df_train_data, df_train_targets, vocabulary_size, load_vocab_file, tf_file, bow_vocab_file)
    X_train = bag_of_words
    X_val = transform_features_noseq(vec_model, df_validate_data["clean_comments"])
    X_test = transform_features_noseq(vec_model, df_test_data["clean_comments"])
    y_val = df_validate_targets['label'].values
    y_train = df_train_targets['label'].values
    y_test = df_test_targets['label'].values
    return X_train, y_train, X_val, y_val, X_test, y_test


def replace_oov_words(data, vocab, unk="<unk>"):
    new_data=[]
    for seq in data:
        new_seq=[]
        for word in seq:
            if word in vocab:
                new_seq.append(word)
            else: new_seq.append(unk)
        new_data.append(new_seq)
    return new_data


def map_words_to_indices(data, vocab):
    new_data = []
    for seq in data:
        new_seq=[]
        for word in seq:
            new_seq.append(vocab[word])
        new_data.append(new_seq)
    return new_data


def create_seq_features(vocab_file, vocabulary_size, remove_stopwords,max_seq_length,padding='post'):
    vocab = cPickle.load(open(vocab_file))
    train_file_cleaned, validate_file_cleaned, test_file_cleaned, _, _ = get_file_names(
        remove_stopwords, vocabulary_size)
    df_train_data, df_train_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + train_file_cleaned)
    df_validate_data, df_validate_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + validate_file_cleaned)
    df_test_data, df_test_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + test_file_cleaned)

    train_data = df_train_data["clean_comments"].tolist()
    val_data = df_validate_data["clean_comments"].tolist()
    test_data = df_test_data["clean_comments"].tolist()
    train_data = replace_oov_words(train_data, vocab, unk="<unk>")
    val_data = replace_oov_words(val_data, vocab, unk="<unk>")
    test_data = replace_oov_words(test_data, vocab, unk="<unk>")

    #add unk symbol to vocab
    vocab["<unk>"]=len(vocab)
    for key,value in vocab.items():
        if value==0:print key
    #start vocab from 1 to account for masking
    vocab={key:value+1 for key,value in vocab.items()}
    for key,value in vocab.items():
        if value==0:print key

    train_data=map_words_to_indices(train_data, vocab)
    val_data=map_words_to_indices(val_data, vocab)
    test_data=map_words_to_indices(test_data, vocab)

    x_train = sequence.pad_sequences(train_data, maxlen=max_seq_length, padding=padding)
    x_val = sequence.pad_sequences(val_data, maxlen=max_seq_length, padding=padding)
    x_test = sequence.pad_sequences(test_data, maxlen=max_seq_length, padding=padding)
    y_train = np.reshape(df_train_targets.values,newshape=(df_train_targets.values.shape[0],))
    y_val = np.reshape(df_validate_targets.values,newshape=(df_validate_targets.values.shape[0],))
    y_test = np.reshape(df_test_targets.values,newshape=(df_test_targets.values.shape[0],))
    return x_train,y_train,x_val,y_val,x_test,y_test


def save_features_and_labels(x_data, y_data, x_path, y_path):
    cPickle.dump(x_data, open(x_path, 'wb'))
    cPickle.dump(y_data, open(y_path, 'wb'))


def features_pipeline(vocabulary_size=5000, max_seq_length=150, clean_data=False, out_folder="experiments/data", subset_size=50000, remove_stopwords=True):
    #clean data and make test and train/val splits. All splits are saved to files
    if clean_data:
        preprocess.clean_and_split_data(subset_size=subset_size, remove_stopwords=remove_stopwords, test_size=0.2, val_size=0.1)
    # Load train/val files and extract train/val features.
    X_train, y_train, X_val, y_val, x_test,y_test = create_noseq_features(vocabulary_size=vocabulary_size, load_vocab_file=False, remove_stopwords=remove_stopwords)
    # #Save train/val noseq features to files
    save_features_and_labels(X_train, y_train, os.path.join(out_folder,"x_train_noseq.pickle"), os.path.join(out_folder,"y_train_noseq.pickle"))
    save_features_and_labels(X_val, y_val, os.path.join(out_folder,"x_val_noseq.pickle"), os.path.join(out_folder,"y_val_noseq.pickle"))
    save_features_and_labels(x_test, y_test, os.path.join(out_folder, "x_test_seq.pickle"),
                             os.path.join(out_folder, "y_test_seq.pickle"))
    vocab_file = get_vocab_filename(remove_stopwords=remove_stopwords, vocabulary_size=vocabulary_size)
    X_train, y_train, X_val, y_val,x_test,y_test = create_seq_features(vocab_file=vocab_file, vocabulary_size=vocabulary_size,
                                                                       max_seq_length=max_seq_length, remove_stopwords=remove_stopwords, padding='post')
    save_features_and_labels(X_train, y_train, os.path.join(out_folder, "x_train_seq.pickle"),
                             os.path.join(out_folder,"y_train_seq.pickle"))
    save_features_and_labels(X_val, y_val, os.path.join(out_folder, "x_val_seq.pickle"), os.path.join(out_folder, "y_val_seq.pickle"))
    save_features_and_labels(x_test, y_test, os.path.join(out_folder, "x_test_seq.pickle"),
                             os.path.join(out_folder, "y_test_seq.pickle"))

    print("Final sequential features were saved in folder %s in files: %s,%s,%s"%
          (out_folder,"x_train_noseq.pickle","x_val_noseq.pickle","x_test_seq.pickle"))
    print("Final non sequential features were saved in folder %s in files: %s,%s,%s" % (out_folder,
                                                                                        "x_train_seq.pickle","x_val_seq.pickle",
                                                                          "x_test_seq.pickle"))

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--clean_data", type="bool", default=False)
    parser.add_argument("--subset_size", type=str, default=None)
    parser.add_argument("--remove_stopwords", type="bool", default=False)
    parser.add_argument("--vocabulary_size", type=int, default=10000)
    parser.add_argument("--max_seq_length", type=int, default=150)
    parser.add_argument("--out_folder", type=str, default=None)


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    params, unparsed = parser.parse_known_args()
    features_pipeline(clean_data=params.clean_data, subset_size=int(params.subset_size) if params.subset_size else None,
                      remove_stopwords=params.remove_stopwords , vocabulary_size=params.vocabulary_size,
                      max_seq_length=params.max_seq_length, out_folder=params.out_folder)

if __name__ == '__main__':
    main()


