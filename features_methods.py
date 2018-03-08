import cPickle
import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
import os
import argparse
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText
import itertools
import nltk


def words_frequencies(texts):
    freq_dist = nltk.FreqDist(itertools.chain(*texts))
    return freq_dist

def build_vocabulary(vocab_freqs, max_features=None, embeddings_map=None):
    new_vocab_freqs={}
    if embeddings_map is not None:
        for word in vocab_freqs:
            if word in embeddings_map: new_vocab_freqs[word]=vocab_freqs[word]
    else: new_vocab_freqs = vocab_freqs
    sorted_words = dict(sorted(new_vocab_freqs.items(), key=lambda x: x[1])).keys()
    if max_features is not None:
        =sorted_words[:max_features]
    vocab={key:i for key,i in enumerate(sorted_words)}
    return vocab

def fit_features_noseq(df_train_data, df_train_targets, vocab, tf_save_file):
    vec_model = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, vocabulary=vocab)
    bag_of_words, vec_model = preprocess.transform_to_vec_values(vec_model, df_train_data, df_train_targets,
                            "clean_comments", 'train_with_tfidf.csv', tf_save_file)
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


def create_noseq_features(df_train_data, df_train_targets, df_validate_data, df_validate_targets, df_test_data,
                          df_test_targets, vocab, tf_save_file):
    bag_of_words, vec_model = fit_features_noseq(df_train_data, df_train_targets, vocab, tf_save_file)
    X_train = bag_of_words
    X_val = transform_features_noseq(vec_model, df_validate_data)
    X_test = transform_features_noseq(vec_model, df_test_data)
    y_train = to_categorical(df_train_targets.values)
    y_val = to_categorical(df_validate_targets.values)
    y_test = to_categorical(df_test_targets.values)
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
    y_train = to_categorical(df_train_targets.values)
    y_val = to_categorical(df_validate_targets.values)
    y_test = to_categorical(df_test_targets.values)
    return x_train,y_train,x_val,y_val,x_test,y_test


def save_features_and_labels(x_data, y_data, x_path, y_path):
    cPickle.dump(x_data, open(x_path, 'wb'))
    cPickle.dump(y_data, open(y_path, 'wb'))


def load_to_gensim(filepath):
    model = KeyedVectors.load_word2vec_format(filepath, binary=False) #GloVe Model - not updatable
    return model


def create_glove_embeddings(words, embeddings_path):
    word2vec = load_to_gensim(embeddings_path)
    glove_word_dict = dict()
    for w in words:
        if w in word2vec.vocab:
            glove_word_dict[w] = word2vec[w]
    print "total words not found in Glove:", len(words) - len(glove_word_dict)
    return glove_word_dict


def create_fasttext_embeddings(words, embeddings_path):
    ff_model = FastText.load_binary_data(embeddings_path)
    glove_word_dict = dict()
    for w in words:
        glove_word_dict[w] =ff_model[w]
    print "total words not found in Glove:", len(words) - len(glove_word_dict)
    return glove_word_dict


def create_embeddings(words, glove_path, fasttext_path):
    glove_map=create_glove_embeddings(words, glove_path)
    fast_text_map=create_fasttext_embeddings(words, fasttext_path)
    return glove_map, fast_text_map


def features_pipeline(vocabulary_size=5000, max_seq_length=150, clean_data=False, out_folder="experiments/data", subset_size=50000, remove_stopwords=True):
    #clean data and make test and train/val splits. All splits are saved to files
    if clean_data:
        preprocess.clean_and_split_data(subset_size=subset_size, remove_stopwords=remove_stopwords, test_size=0.2, val_size=0.1)

    # load cleaned dfs
    train_file_cleaned, validate_file_cleaned, test_file_cleaned, tf_file, bow_vocab_file = get_file_names(
        remove_stopwords, vocabulary_size)
    df_train_data, df_train_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + train_file_cleaned)
    df_validate_data, df_validate_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + validate_file_cleaned)
    df_test_data, df_test_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + test_file_cleaned)

    # Get a map of words to frequencies based on train data.
    vocab_freqs = words_frequencies(df_train_data['clean_comments'])
    # Get a map of words to embeddings based on train data.
    glove_map, fast_text_map = create_embeddings(vocab_freqs.keys(), "embeddings/100d_glove_english_only.txt", "embeddings/")
    # Get a vocabulary of words to indices based on train data and on word embeddings.
    vocab_glove=build_vocabulary(glove_map, embeddings_map=glove_map, max_features=vocabulary_size)
    vocab_fasttext = build_vocabulary(fast_text_map, embeddings_map=fast_text_map, max_features=vocabulary_size)
    vocab_full = build_vocabulary(fast_text_map, embeddings_map=None, max_features=vocabulary_size)

    # create non sequential bow and tf-idf features, given a vocab file. No embeddings are involved in these features.
    x_train, y_train, x_val, y_val, x_test,y_test = create_noseq_features(df_train_data['clean_comments'], df_train_targets['label'],
                                                                          df_validate_data['clean_comments'], df_validate_targets['label'],
                                                                          df_test_data['clean_comments'],
                                                                            df_test_targets['label'], vocab_full, tf_file)
    # #Save train/val noseq bow features to files
    save_features_and_labels(x_train, y_train, os.path.join(out_folder,"x_train_bow_noseq.pickle"), os.path.join(out_folder,"y_train_bow_noseq.pickle"))
    save_features_and_labels(x_val, y_val, os.path.join(out_folder,"x_val_bow_noseq.pickle"), os.path.join(out_folder,"y_val_bow_noseq.pickle"))
    save_features_and_labels(x_test, y_test, os.path.join(out_folder, "x_test_bow_seq.pickle"),
                             os.path.join(out_folder, "y_test_bow_seq.pickle"))
    print("Non sequential shapes: ")
    print("x_train,y_train shapes: (%s,%s)"%(x_train.shape,y_train.shape))
    print("x_val,y_val shapes: (%s,%s)" % (x_val.shape, y_val.shape))
    print("x_test,y_test shapes: (%s,%s)" % (x_test.shape, y_test.shape))


    vocab_file = get_vocab_filename(remove_stopwords=remove_stopwords, vocabulary_size=vocabulary_size)
    x_train, y_train, x_val, y_val,x_test,y_test = create_seq_features(vocab_file=vocab_file, vocabulary_size=vocabulary_size,
                                                                       max_seq_length=max_seq_length, remove_stopwords=remove_stopwords, padding='post')
    save_features_and_labels(x_train, y_train, os.path.join(out_folder, "x_train_seq.pickle"),
                             os.path.join(out_folder,"y_train_seq.pickle"))
    save_features_and_labels(x_val, y_val, os.path.join(out_folder, "x_val_seq.pickle"), os.path.join(out_folder, "y_val_seq.pickle"))
    save_features_and_labels(x_test, y_test, os.path.join(out_folder, "x_test_seq.pickle"),
                             os.path.join(out_folder, "y_test_seq.pickle"))

    print("Sequential shapes: ")
    print("x_train,y_train shapes: (%s,%s)" % (x_train.shape, y_train.shape))
    print("x_val,y_val shapes: (%s,%s)" % (x_val.shape, y_val.shape))
    print("x_test,y_test shapes: (%s,%s)" % (x_test.shape, y_test.shape))

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


