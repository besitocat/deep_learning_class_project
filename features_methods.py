import cPickle
print("%s"%str(None))
import preprocess
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from keras.preprocessing import sequence
from keras.utils import to_categorical
import argparse
from gensim.models.fasttext import FastText
import itertools
import nltk
import numpy as np
import os

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def create_glove_embeddings(words, embeddings_path):
    print("Loading Glove embeddings...")
    word2vec = preprocess.load_to_gensim(embeddings_path)
    print("Total embeddings loaded: %d"%len(word2vec.vocab))
    glove_word_dict = dict()
    for w in words:
        if w in word2vec.vocab:
            glove_word_dict[w] = word2vec[w]
    print "total words not found in Glove: %d out of %d"%(len(words) - len(glove_word_dict),len(words))
    return glove_word_dict


def create_fasttext_embeddings(words, embeddings_path):
    print("Loading fasttext model...")
    ff_model = FastText.load_fasttext_format(embeddings_path)
    fasttext_word_dict = dict()
    for w in words:
        try:
            fasttext_word_dict[w] =ff_model[w]
        except:
            pass
    print "total words not found in Fasttext:", len(words) - len(fasttext_word_dict)
    return fasttext_word_dict


def replace_oov_words(data, vocab, unk="<unk>"):
    new_data = []
    for seq in data:
        new_seq = []
        for word in seq:
            if word in vocab:
                new_seq.append(word)
            else:
                new_seq.append(unk)
        new_data.append(new_seq)
    return new_data


def map_words_to_indices(data, vocab):
    new_data = []
    for seq in data:
        new_seq = []
        for word in seq:
            new_seq.append(vocab[word])
        new_data.append(new_seq)
    return new_data


def words_frequencies(texts):
    freq_dist = nltk.FreqDist(itertools.chain(*texts))
    return freq_dist


def build_vocabulary(vocab_freqs, max_features=None, min_freq=None, embeddings_map=None):
    if max_features: print("Building vocabulary with max_features %d"%max_features)
    if min_freq: print("Building vocabulary with word frequencies greater than %d" % min_freq)
    new_vocab_freqs={}
    if embeddings_map is not None:
        for word in vocab_freqs:
            if word in embeddings_map: new_vocab_freqs[word]=vocab_freqs[word]
    else: new_vocab_freqs = vocab_freqs
    sorted_words = sorted(new_vocab_freqs.items(), key=lambda x: x[1])
    start_idx=0
    if min_freq is not None:
        for (word,freq) in sorted_words:
            if freq<min_freq: start_idx+=1
    sorted_words=sorted_words[start_idx:]
    if max_features is not None:
        sorted_words=sorted_words[-max_features:]
    sorted_words = dict(sorted_words).keys()
    vocab={word:i for i,word in enumerate(sorted_words)}
    return vocab


def build_embeddings_vocabulary(vocab_freqs, embeddings_map, min_freq=None, max_features=None):
    if max_features: print("Building vocabulary with max_features %d"%max_features)
    if min_freq: print("Building vocabulary with word frequencies greater than %d" % min_freq)
    new_vocab_freqs={}
    for word in vocab_freqs:
        if word in embeddings_map: new_vocab_freqs[word]=vocab_freqs[word]
    sorted_words = sorted(new_vocab_freqs.items(), key=lambda x: x[1])
    start_idx = 0
    if min_freq is not None:
        for (word, freq) in sorted_words:
            if freq < min_freq: start_idx += 1
    sorted_words = sorted_words[start_idx:]
    if max_features is not None:
        sorted_words=sorted_words[-max_features:]
    sorted_words = dict(sorted_words).keys()
    vocab={word:embeddings_map[word] for i,word in enumerate(sorted_words)}
    return vocab


def transform_to_vec_values(vec_model, df, column_name, tfidf_save_file):
    print "\n**** Transforming to numerical values.... to file: ", tfidf_save_file
#    vec_model = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, use_idf=False)
    bag_of_words = vec_model.fit_transform(df[column_name])
    print "shape of bow transformation:", bag_of_words.shape
    #save to pickle file
    # cPickle.dump(bag_of_words, open(preprocess.root_sarcasm_data_dir + tf_save_file, 'wb'))#"bag_of_words_values_"+#str(vocabulary_size)+".pkl", 'wb'))
    print "**** Transforming to TF-IDF ****"
    tfidf_model = TfidfTransformer()
    tfidf_values = tfidf_model.fit_transform(bag_of_words)
    print "shape of tf-idf transformation:", tfidf_values.shape

    # print "**** Adding labels ****"
    # labels = list(df_labels['label'])
    # df.insert(loc=0, column='label', value=labels)
    # df['tf-idf-transform'] = list(tfidf_values)
    # df.to_csv(preprocess.root_sarcasm_data_dir + tfidf_save_file)
    # print "**** Transforming to TF-IDF is COMPLETE. New column - tf-idf-transform added to file: " \
    #       + tfidf_save_file + " and the vectorized was saved into a pickle file"
    return bag_of_words, tfidf_values, vec_model, tfidf_model

def create_noseq_features(df_train_data, df_validate_data, df_test_data, column_name, vocab):
    vec_model = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, vocabulary=vocab)
    bag_of_words, tfidf_values, vec_model, tfidf_model = transform_to_vec_values(vec_model, df_train_data,
                                                                                 column_name, 'train_with_tfidf.csv',)
    x_train_bow = bag_of_words
    x_val_bow = vec_model.transform(df_validate_data[column_name])
    x_test_bow = vec_model.transform(df_test_data[column_name])

    x_train_tfidf = tfidf_values
    x_val_tfidf = tfidf_model.transform(x_val_bow)
    x_test_tfidf = tfidf_model.transform(x_test_bow)
    return x_train_bow, x_val_bow, x_test_bow, x_train_tfidf, x_val_tfidf, x_test_tfidf


def create_noseq_embed_features(df_train_data, df_validate_data, df_test_data,
                              column_name, embedding_map):
    vec_model=MeanEmbeddingVectorizer(embedding_map)
    x_train = vec_model.transform(df_train_data[column_name])
    x_val = vec_model.transform(df_validate_data[column_name])
    x_test = vec_model.transform(df_test_data[column_name])
    return x_train, x_val, x_test


def create_labels(df_train_targets,df_validate_targets,df_test_targets):
    y_train = to_categorical(df_train_targets.values)
    y_val = to_categorical(df_validate_targets.values)
    y_test = to_categorical(df_test_targets.values)
    return y_train,y_val,y_test


def create_seq_features(df_train_data, df_validate_data, df_test_data,
                              column_name, vocab,max_seq_length,padding='post', embedding_map=None):
    train_data = df_train_data[column_name].tolist()
    val_data = df_validate_data[column_name].tolist()
    test_data = df_test_data[column_name].tolist()
    train_data = replace_oov_words(train_data, vocab, unk="<unk>")
    val_data = replace_oov_words(val_data, vocab, unk="<unk>")
    test_data = replace_oov_words(test_data, vocab, unk="<unk>")

    train_data=map_words_to_indices(train_data, vocab)
    val_data=map_words_to_indices(val_data, vocab)
    test_data=map_words_to_indices(test_data, vocab)

    # build embeddings matrix
    embedding_matrix=None
    if embedding_map is not None:
        emb_dim = len(embedding_map.itervalues().next())
        embedding_map["<unk>"]=np.zeros(shape = (emb_dim,))
        embedding_matrix = np.zeros((len(vocab)+1, emb_dim))
        for word,idx in vocab.items():
            embedding_matrix[idx] = embedding_map[word]
    x_train = sequence.pad_sequences(train_data, maxlen=max_seq_length, padding=padding)
    x_val = sequence.pad_sequences(val_data, maxlen=x_train.shape[1], padding=padding)
    x_test = sequence.pad_sequences(test_data, maxlen=x_train.shape[1], padding=padding)
    return x_train,x_val,x_test,embedding_matrix


def features_pipeline(vocabulary_size=5000, max_seq_length=150, clean_data=False, out_folder="experiments/data",
                      subset_size=50000, min_freq=None, remove_stopwords=True, fast_text=False, glove=True,
                      glove_map_file=None, fast_text_map_file=None, seq_features=True, glove_file="embedding/100d_glove_english_only.txt"):
    noseq_prefix=str(subset_size)+"_"+str(vocabulary_size)+"_"+str(min_freq)+"_"
    seq_prefix = str(subset_size) + "_" + str(vocabulary_size) + "_" + str(min_freq) + "_"+str(max_seq_length) + "_"
    import cPickle
    #clean data and make test and train/val splits. All splits are saved to files
    if clean_data:
        preprocess.clean_and_split_data(subset_size=subset_size, remove_stopwords=remove_stopwords, test_size=0.2, val_size=0.1)

    # load cleaned dfs
    train_file_cleaned, validate_file_cleaned, test_file_cleaned, tf_file, bow_vocab_file = preprocess.get_file_names(
        remove_stopwords, vocabulary_size)
    df_train_data, df_train_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + train_file_cleaned)
    df_validate_data, df_validate_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + validate_file_cleaned)
    df_test_data, df_test_targets = preprocess.load_preprocessed_file(
        preprocess.root_sarcasm_data_dir + test_file_cleaned)
    print("Loaded cleaned data: %s,%s,%s"%(preprocess.root_sarcasm_data_dir + train_file_cleaned,
                                           preprocess.root_sarcasm_data_dir + validate_file_cleaned,
                                           preprocess.root_sarcasm_data_dir + test_file_cleaned))
    print("Train: %d, Validation: %d, Test: %d"%(len(df_train_data),len(df_validate_data),len(df_test_data)))
    print("\n")

    df_train_targets['label'] = df_train_targets['label'].apply(lambda x: x-1.0)
    df_validate_targets['label'] = df_validate_targets['label'].apply(lambda x: x - 1.0)
    df_test_targets['label'] = df_test_targets['label'].apply(lambda x: x - 1.0)
    print("Creating one-hot labels...")

    y_train, y_val, y_test = create_labels(df_train_targets, df_validate_targets, df_test_targets)
    preprocess.save_labels(y_train, y_val, y_test, out_folder)

    # Create a map of words to frequencies based on train data.
    vocab_freqs = words_frequencies(df_train_data['clean_comments'])
    if remove_stopwords:
        print("Total unique words in corpus (excluding stopwords): %d"%(len(vocab_freqs)))
    else:
        print("Total unique words in corpus (including stopwordS): %d" % (len(vocab_freqs)))
    print("Our vocab size will be: %d"%vocabulary_size if vocabulary_size is not None else len(vocab_freqs))
    print("\n")
    # Create or load a map of words to embeddings based on train data.
    glove_map=None
    if glove:
        if glove_map_file is None:
            glove_map = create_glove_embeddings(vocab_freqs.keys(), glove_file)
            print("Saving glove")
            cPickle.dump(glove_map, open(os.path.join(out_folder,"glove_map"+str(subset_size)+".pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            glove_map = cPickle.load(open(glove_map_file))
        print("Glove map of size: %d"%len(glove_map))
    fast_text_map=None
    if fast_text:
        if fast_text_map_file is None:
            fast_text_map = create_fasttext_embeddings(vocab_freqs.keys(), "embedding/wiki.simple")
            print("Saving fasttext")
            cPickle.dump(fast_text_map, open(os.path.join(out_folder,"fast_text_map"+str(subset_size)+".pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            fast_text_map = cPickle.load(open(fast_text_map_file))
        print("Fasttext map of size: %d" % len(fast_text_map))

    # Get a map of words to indices based on train data and on word embeddings.
    # We first remove from vocab_freqs the words that do not appear in the embeddings map. Then we select the max_features words.
    # Glove vocabulary
    vocab_glove = None
    if glove:
        vocab_glove = build_embeddings_vocabulary(vocab_freqs, embeddings_map=glove_map, min_freq=min_freq, max_features=vocabulary_size)
        cPickle.dump(vocab_glove,open(os.path.join(out_folder,noseq_prefix+"vocab_glove.pickle"),"wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        print("Vocab size when glove embeddings: %d"%len(vocab_glove))
    # Fasttext vocabulary. Careful! It is different thatn Glove vocab, cause with fasttext we have an embeddings for all train words.
    vocab_fasttext = None
    if fast_text:
        vocab_fasttext = build_embeddings_vocabulary(vocab_freqs, embeddings_map=fast_text_map, min_freq=min_freq, max_features=vocabulary_size)
        cPickle.dump(vocab_fasttext, open(os.path.join(out_folder,noseq_prefix+"vocab_fasttext.pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        print("Vocab size when fasttext embeddings: %d" % len(vocab_fasttext))
    # Full vocabulary. We select max_features words. We DO NOT remove words that do not appear to some embeddings matrix.
    vocab_full = build_vocabulary(vocab_freqs, embeddings_map=None, min_freq=min_freq, max_features=vocabulary_size)
    cPickle.dump(vocab_full, open(os.path.join(out_folder,"vocab_full.pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
    print("Full vocab size: %d" % len(vocab_full))
    print("\n")

    # create and save non sequential bow and tf-idf features, given the full vocabulary. No embeddings are involved in these features.
    x_train_bow, x_val_bow, x_test_bow, x_train_tfidf, x_val_tfidf, x_test_tfidf\
        = create_noseq_features(df_train_data,df_validate_data,df_test_data, 'clean_comments',vocab_full)
    print("Saving bow features...")
    preprocess.save_features(x_train_bow, x_val_bow, x_test_bow, out_folder, suffix=noseq_prefix+"bow_noseq")
    print("Saving tf-idf features...")
    preprocess.save_features(x_train_tfidf, x_val_tfidf, x_test_tfidf, out_folder, suffix=noseq_prefix+"tfidf_noseq")

    # create and save non sequential features from glove embeddings vocab.
    if glove:
        x_train, x_val, x_test = create_noseq_embed_features(df_train_data,df_validate_data,df_test_data,'clean_comments',
                                                                                    vocab_glove)
        print("Saving glove features...")
        preprocess.save_features(x_train, x_val, x_test, out_folder, suffix=noseq_prefix+"glove_emb_noseq")


    # create and save non sequential features from fasttext embeddings vocab.
    if fast_text:
        x_train, x_val, x_test = create_noseq_embed_features(df_train_data,df_validate_data,df_test_data,'clean_comments',
                                                                                    vocab_fasttext)
        print("Saving fasttext features...\n")
        preprocess.save_features(x_train, x_val, x_test, out_folder, suffix=noseq_prefix+"fasttext_emb_noseq")


    if seq_features:
        # create and save sequential features
        print("Creating Sequential features:")
        extend_vocab_for_seq(vocab_full)
        cPickle.dump(vocab_full,
                     open(os.path.join(out_folder,"sequential/" + seq_prefix + "vocab_full_seq.pickle"), "wb"),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        x_train, x_val, x_test, embedding_matrix = \
            create_seq_features(df_train_data, df_validate_data,
                                df_test_data, 'clean_comments', vocab=vocab_full,
                                max_seq_length=max_seq_length, padding='post', embedding_map=None)
        preprocess.save_features(x_train, x_val, x_test, out_folder+"/sequential/", suffix=seq_prefix+"noemb_seq")
        if glove:
            vocab_glove_seq={word: i for i, word in enumerate(vocab_glove)}
            extend_vocab_for_seq(vocab_glove_seq)
            cPickle.dump(vocab_glove_seq, open(os.path.join(out_folder,"sequential/"+seq_prefix+"vocab_glove_seq.pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
            x_train, x_val,x_test,embedding_matrix = \
                create_seq_features(df_train_data, df_validate_data,
                                    df_test_data, 'clean_comments', vocab=vocab_glove_seq,
                                    max_seq_length=max_seq_length, padding='post', embedding_map=vocab_glove)
            preprocess.save_features(x_train, x_val, x_test, out_folder+"/sequential/", suffix=seq_prefix+"glove_seq")
            cPickle.dump(embedding_matrix, open(os.path.join(out_folder,"sequential/"+seq_prefix+"glove_matrix.pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
            print("Glove matrix shape: ",embedding_matrix.shape)
        if fast_text:
            vocab_fasttext_seq = {word: i for i, word in enumerate(vocab_fasttext)}
            extend_vocab_for_seq(vocab_fasttext_seq)
            cPickle.dump(vocab_fasttext_seq, open(os.path.join(out_folder,"sequential/"+seq_prefix+"vocab_fasttext_seq.pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
            x_train, x_val, x_test, embedding_matrix = \
                create_seq_features(df_train_data, df_validate_data,
                                    df_test_data, 'clean_comments', vocab={word:i for i,word in enumerate(vocab_fasttext)},
                                    max_seq_length=max_seq_length, padding='post', embedding_map=vocab_fasttext_seq)
            preprocess.save_features(x_train, x_val, x_test, out_folder+"/sequential/", suffix=seq_prefix+"fast_text_seq")
            cPickle.dump(embedding_matrix, open(os.path.join(out_folder,"sequential/"+seq_prefix+"fasttext_matrix.pickle"), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
            print("Fasttext matrix shape: ", embedding_matrix.shape)
            print("\n")
            print("Done creating features.")
            print("Subset size: %s,Max vocab size: %s, Min freq: %s, Max seq length: %s"%
                  (str(subset_size),str(vocabulary_size),str(min_freq),str(max_seq_length)))

def extend_vocab_for_seq(vocab):
    # add unk symbol to vocab
    vocab["<unk>"] = len(vocab)
    # start vocab from 1 to account for masking
    vocab.update((key,value + 1) for key, value in vocab.items())

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--clean_data", type="bool", default=False)
    parser.add_argument("--subset_size", type=str, default=None)
    parser.add_argument("--remove_stopwords", type="bool", default=False)
    parser.add_argument("--vocabulary_size", type=str, default=None)
    parser.add_argument("--min_freq", type=str, default=None)
    parser.add_argument("--max_seq_length", type=str, default=None)
    parser.add_argument("--out_folder", type=str, default=None)
    parser.add_argument("--glove", type="bool", default=True)
    parser.add_argument("--fast_text", type="bool", default=True)
    parser.add_argument("--fast_text_map_file", type=str, default=None)
    parser.add_argument("--glove_map_file", type=str, default=None)
    parser.add_argument("--seq_features", type="bool", default=True)
    parser.add_argument("--is_yelp", type="bool", default=True)
    parser.add_argument("--glove_file", type=str, default="embedding/100d_glove_english_only.txt")
def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    params, unparsed = parser.parse_known_args()
    if params.is_yelp:
        preprocess.prepare_for_yelp(is_yelp=True)
    features_pipeline(clean_data=params.clean_data, subset_size=int(params.subset_size) if params.subset_size else None,
                      remove_stopwords=params.remove_stopwords , vocabulary_size=int(params.vocabulary_size) if params.vocabulary_size else None,
                      min_freq=int(params.min_freq) if params.min_freq else None,
                      max_seq_length=int(params.max_seq_length) if params.max_seq_length else None, out_folder=params.out_folder,
                      glove_map_file=params.glove_map_file, fast_text_map_file=params.fast_text_map_file,
                      glove=params.glove, fast_text=params.fast_text, seq_features=params.seq_features, glove_file=params.glove_file)

if __name__ == '__main__':
    main()