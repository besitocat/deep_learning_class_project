from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad
import model
import os
import utils
import time

def train_model(model, x_train, y_train, out_dir,  validation_data, n_epochs, batch_size, learning_rate,
                loss="binary_crossentropy", early_stopping=True, save_checkpoint=True, verbose=1):
    callbacks = []
    if save_checkpoint:
        # save the model at every epoch. 'val_loss' is the monitored quantity.
        # If save_best_only=True, the model with the best monitored quantity is not overwitten.
        # If save_weights_only=True, only the model weights are saved calling the method model.save_weights
        checkpoint = ModelCheckpoint(os.path.join(out_dir,model.model_name + ".{epoch:02d}-{val_loss:.3f}.hdf5"),
                                          verbose=verbose, monitor='val_loss', save_weights_only=True, save_best_only=True)
        callbacks.append(checkpoint)
    if early_stopping:
        # Training stops when the monitored quantity (val_loss) stops improving.
        # patience is the number of epochs with no improvement after which training is stopped.
        stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=15, verbose=verbose, mode='auto')
        callbacks.append(stopping)
    adam = Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0, clipnorm=1.)
    model.compile(metrics=[], optimizer=adam, loss=loss)
    print("Training of model '%s' started."%model.model_name)
    start_time = time.time()
    history = model.fit(x_train, y_train, validation_data=validation_data, n_epochs=n_epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    print("Training of model '%s' finished in %s." % (model.model_name,time.strftime("%H:%M:%S",
                                                                                         time.gmtime(time.time()-start_time))))
    return history


def train_ffn_model(x_train, y_train, x_val, y_val, params):
    fnn_model = model.FFNModel(input_size=params["input_size"], output_size=params["output_size"],
                               model_name=params["model_name"],
                               hidden_activation=params["hidden_activation"], out_activation=params["out_activation"],
                               hidden_dims=params["hidden_dims"],
                               layers=params["layers"], kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"],
                               dropouts=params["dropouts"])
    history = train_model(fnn_model, x_train, y_train, params["out_dir"], validation_data=(x_val, y_val), save_checkpoint=params["save_checkpoint"],
                n_epochs=params["n_epochs"], batch_size=params["batch_size"], verbose=params["verbose"],
                early_stopping=params["early_stopping"], learning_rate=params["learning_rate"], loss=params["loss"])
    return utils.extract_results_from_history(history)


def train_rnn_model(x_train, y_train, x_val, y_val, params):
    rnn_model = model.RNNModel(max_seq_length=params["max_seq_length"], input_size=params["input_size"],
                               output_size=params["output_size"], embed_dim=params["embed_dim"],
                               emb_trainable=params["emb_trainable"], model_name=params["model_name"],
                               hidden_activation=params["hidden_activation"], out_activation=params["out_activation"],
                               hidden_dim=params["hidden_dims"][0], kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"], recurrent_regularizer=params["recurrent_regularizer"],
                               input_dropout=params["input_dropout"], recurrent_dropout=params["recurrent_dropout"],
                               rnn_unit_type=params["rnn_unit_type"],  bidirectional=params["bidirectional"])
    history = train_model(rnn_model, x_train, y_train, out_dir=params["out_dir"], validation_data=(x_val, y_val), save_checkpoint=params["save_checkpoint"],
                          n_epochs=params["n_epochs"], batch_size=params["batch_size"], verbose=params["verbose"],
                          early_stopping=params["early_stopping"], learning_rate=params["learning_rate"], loss=params["loss"])
    return utils.extract_results_from_history(history)


def train_rnn(params):
    x_train, y_train = utils.load_data(params["x_train_path"], params["y_train_path"])
    x_val, y_val = utils.load_data(params["x_val_path"], params["y_val_path"])
    train_rnn_model(x_train, y_train, x_val, y_val, params)


def train_ffn(params):
    x_train, y_train = utils.load_data(params["x_train_path"], params["y_train_path"])
    x_val, y_val = utils.load_data(params["x_val_path"], params["y_val_path"])
    train_ffn_model(x_train, y_train, x_val, y_val, params)
