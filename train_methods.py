from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adagrad
import model
import os
import utils


def train_model(model, x_train, y_train, validation_data, n_epochs, batch_size, learning_rate,
                early_stopping=True, save_checkpoint=True, verbose=1, out_dir="trained_models"):
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
    model.compile(metrics=[], optimizer=adam)
    history = model.fit(x_train, y_train, validation_data=validation_data, n_epochs=n_epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    return history


def train_ffn_model(x_train, y_train, x_val, y_val, input_size, output_size, model_name, hidden_activation, out_activation,
                 hidden_dims, layers, kernel_initializer, kernel_regularizer, dropouts,
                    n_epochs, batch_size, learning_rate, save_checkpoint, early_stopping, verbose):
    fnn_model = model.FFNModel(input_size=input_size, output_size=output_size, model_name=model_name,
                               hidden_activation=hidden_activation, out_activation=out_activation, hidden_dims=hidden_dims,
                               layers=layers, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                               dropouts=dropouts)
    history = train_model(fnn_model, x_train, y_train, validation_data=(x_val, y_val), save_checkpoint=save_checkpoint,
                n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                early_stopping=early_stopping, learning_rate=learning_rate)
    return utils.extract_results_from_history(history)


def train_rnn_model(x_train, y_train, x_val, y_val, max_seq_length, input_size, output_size, model_name, hidden_activation,
                 out_activation, hidden_dim, kernel_initializer, kernel_regularizer,
                 recurrent_regularizer, input_dropout, recurrent_dropout,
                 rnn_unit_type, bidirectional, embed_dim, emb_trainable, n_epochs, batch_size, learning_rate, save_checkpoint,
                    early_stopping, verbose):
    rnn_model = model.RNNModel(max_seq_length=max_seq_length, input_size=input_size, output_size=output_size, embed_dim=embed_dim, emb_trainable=emb_trainable,
                               model_name=model_name, hidden_activation=hidden_activation, out_activation=out_activation, hidden_dim=hidden_dim,
                               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                               recurrent_regularizer=recurrent_regularizer, input_dropout=input_dropout,
                               recurrent_dropout=recurrent_dropout, rnn_unit_type=rnn_unit_type,  bidirectional=bidirectional)
    history = train_model(rnn_model, x_train, y_train, validation_data=(x_val, y_val), save_checkpoint=save_checkpoint,
                          n_epochs=n_epochs, batch_size=batch_size, verbose=verbose,
                          early_stopping=early_stopping, learning_rate=learning_rate)
    return utils.extract_results_from_history(history)


def train_rnn(x_train_path, y_train_path, x_val_path, y_val_path,
          max_seq_length, input_size, output_size, model_name="rnn_model", hidden_activation="relu",
          out_activation="sigmoid", hidden_dim=32, kernel_initializer="uniform", kernel_regularizer=None,
          recurrent_regularizer=None, input_dropout=0.0, recurrent_dropout=0.0,
          rnn_unit_type="rnn", bidirectional=False, embed_dim=32, emb_trainable=False, n_epochs=10, batch_size=128,
              learning_rate=0.01, save_checkpoint=True, early_stopping=True, verbose=1):
    x_train, y_train = utils.create_seq_input_data(x_train_path, y_train_path, max_seq_length)
    x_val, x_val = utils.create_seq_input_data(x_val_path, y_val_path, max_seq_length)
    train_rnn_model(x_train, y_train, x_val, x_val, max_seq_length=max_seq_length, input_size=input_size, output_size=output_size,
                    model_name=model_name, hidden_activation=hidden_activation,
                    out_activation=out_activation, hidden_dim=hidden_dim, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer, input_dropout=input_dropout, recurrent_dropout=recurrent_dropout,
                    rnn_unit_type=rnn_unit_type, bidirectional=bidirectional, embed_dim=embed_dim, emb_trainable=emb_trainable,
                    n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_checkpoint,
                    early_stopping=early_stopping, verbose=verbose)


def train_ffn(x_train_path, y_train_path, x_val_path, y_val_path,
                  input_size, output_size, model_name="ffn_model", hidden_activation="relu",
                  out_activation="sigmoid", hidden_dims=[32], layers=1, kernel_initializer="uniform", kernel_regularizer=None,
                  dropouts=[0.0], n_epochs=10, batch_size=128, learning_rate=0.01,
                  save_checkpoint=True, early_stopping=True, verbose=1):
    x_train, y_train = utils.create_nonseq_input_data(x_train_path, y_train_path)
    x_val, x_val = utils.create_nonseq_input_data(x_val_path, y_val_path)
    train_ffn_model(x_train, y_train, x_val, x_val, input_size=input_size, output_size=output_size, model_name=model_name,
                    hidden_activation=hidden_activation, out_activation=out_activation,
             hidden_dims=hidden_dims, layers=layers, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, dropouts=dropouts,
                n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_checkpoint, early_stopping=early_stopping, verbose=verbose)
