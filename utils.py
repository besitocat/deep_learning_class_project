import model
from keras.preprocessing import sequence
import numpy as np
from keras.optimizers import Adagrad
import os
import cPickle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'SimHei'
matplotlib.rcParams['font.serif'] = 'SimHei'
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_attention_weights(x_data,  attn_weights, vocab, out_folder, ids, savefilename, pad_idx=0):
    for i,sample in enumerate(x_data):
        sample = sample.tolist()
        weights = attn_weights[i]
        pad_start = len(sample)
        if pad_idx in sample:
            pad_start = sample.index(pad_idx)

        sample = sample[:pad_start]
        for j, idx in enumerate(sample):
            sample[j] = vocab[idx]
        weights = weights[:pad_start]
        fig = plt.figure(figsize=(15, 15))
        plt.imshow(
            X=np.reshape(weights,(1,weights.shape[0])),
            interpolation="nearest",
            cmap=plt.cm.Blues)
        plt.title("")
        plt.xticks(np.arange(len(sample)), sample, rotation=45)
        plt.yticks(np.arange(1), ["input"])
        plt.savefig(os.path.join(out_folder, savefilename + str(ids[i]) + ".png"))
        plt.close()


def save_predictions(predictions, out_folder, filename="predictions.txt"):
    np.savetxt(os.path.join(out_folder,filename),predictions)
    # with open(os.path.join(out_folder,"predictions.txt"),"w") as f:
    #     newline=""
    #     for preds in predictions:
    #         line = ",".join([str(p) for p in preds])
    #         f.write(newline+line)
    #         newline = "\n"


def load_predictions(filepath):
    with open(filepath,"r") as f:
        predictions = np.loadtxt(filepath)
    return predictions


def load_model(ckpt_weights_file, model, learning_rate):
    adam = Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0, clipnorm=1.)
    model.compile(metrics=[], optimizer=adam)
    model.load_model_weights(ckpt_weights_file)


def get_file_name(params):
    suffix=""
    if params["bidirectional"]:
        suffix+="_bid"
    if params["attention"]:
        suffix+="_attn"
    return params["model_name"]+"_"+"insize"+"_"+str(params["input_size"])+"_"+"h_act"+"_"+params["hidden_activation"]\
           +"_"+"out_act"+"_"+params["out_activation"]+"_"+"h_dims"+"_"+\
           str(params["hidden_dims"])+"_"+"layers"+"_"+str(params["layers"])+"_"+"dropouts"+"_"+str(params["dropouts"])+suffix


def extract_results_from_history(history):
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    print("train losses: %s"%train_losses)
    print("val losses: %s" % val_losses)
    print "min val loss: %f at epoch: %d" % (np.min(val_losses), np.argmin(val_losses) + 1)
    print "train loss: %f at epoch: %d" % (train_losses[np.argmin(val_losses)], np.argmin(val_losses) + 1)
    results = model.TrainResults(train_losses[np.argmin(val_losses)], np.min(val_losses), np.argmin(val_losses) + 1)
    return results


def create_test_model_input_data(x_data_path, y_data_path, max_seq_length):
    """ This method was just to test the models on imdb data"""

    # load a subset of imdb reviews to test the code
    with open(x_data_path, "r") as f:
        x_train = cPickle.load(f)
    if y_data_path is not None:
        with open(y_data_path, "r") as f:
            y_train = cPickle.load(f)
    # pad with zeros all sequences in order to have the same length
    # (i.e. the same number of timesteps). The zero timesteps will be ignored by the model.
    x_data = sequence.pad_sequences(x_train, maxlen=max_seq_length)
    y_data = None
    if y_data_path is not None:
        y_data = np.array(y_train)
    return x_data, y_data


def load_pickle_file(data_path):
    return cPickle.load(open(data_path))


def load_data(x_path,y_path):
    x_train = load_pickle_file(x_path)
    y_train = load_pickle_file(y_path)
    return x_train,y_train