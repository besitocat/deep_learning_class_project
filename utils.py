import model
from keras.preprocessing import sequence
import numpy as np
from keras.optimizers import Adagrad
import os
import cPickle


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


def extract_results_from_history(history):
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
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