import train_methods
import eval_methods

#common parameters
input_size = 2000
output_size = 2
model_name = "rnn_test"
hidden_activation = "relu"
out_activation = "sigmoid"
kernel_initializer = "uniform"
kernel_regularizer = None

#common parameters for training
n_epochs = 20
batch_size = 100
learning_rate = 0.01
save_checkpoint = True
early_stopping = True
verbose = 1
loss_type = "binary_crossentropy"

#rnn specific parameters
rnn_unit_type = 'rnn'
hidden_dim = 32
embed_dim = 32
emb_trainable = True
bidirectional = False
max_seq_length = 100
recurrent_regularizer = None
recurrent_dropout = 0.0
input_dropout = 0.0

#ffn specific parameters
hidden_dims = [32]
layers = 1
dropouts = [0.0]

model_type="rnn"
x_train_path = "experiments/data/x_train_subset.pickle"
y_train_path = "experiments/data/y_train_subset.pickle"
x_val_path = "experiments/data/x_test_subset.pickle"
y_val_path = "experiments/data/y_test_subset.pickle"
eval_weights_ckpt = "experiments/trained_models/rnn_test.02-0.667.hdf5"
eval_x_data = "experiments/data/x_test_subset.pickle"
eval_y_data = "experiments/data/y_test_subset.pickle"
eval_res_folder = "experiments/results"

if __name__ == "__main__":

    if eval_weights_ckpt is not None:
        if model_type == "rnn":
            eval_methods.evaluate_rnn(eval_weights_ckpt, eval_res_folder, eval_x_data, eval_y_data, max_seq_length, input_size, output_size, model_name,
                       hidden_activation, out_activation, hidden_dim, kernel_initializer, kernel_regularizer,
                       recurrent_regularizer, rnn_unit_type, bidirectional, embed_dim, emb_trainable, learning_rate,
                       batch_size, verbose)
        elif model_type == "ffn":
            eval_methods.evaluate_ffn(eval_weights_ckpt, eval_res_folder, eval_x_data, eval_y_data, input_size, output_size,
                         model_name, hidden_activation,
                         out_activation, hidden_dims, layers, kernel_initializer, kernel_regularizer, learning_rate,
                         batch_size, verbose)
    else:
        if model_type=="rnn":
            train_methods.train_rnn(x_train_path=x_train_path, y_train_path=y_train_path, x_val_path=x_val_path, y_val_path=y_val_path,
                                    max_seq_length=max_seq_length, input_size=input_size, output_size=output_size,
                                    model_name=model_name, hidden_activation=hidden_activation,
                                    out_activation=out_activation, hidden_dim=hidden_dim, kernel_initializer=kernel_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer, input_dropout=input_dropout, recurrent_dropout=recurrent_dropout,
                                    rnn_unit_type=rnn_unit_type, bidirectional=bidirectional, embed_dim=embed_dim, emb_trainable=emb_trainable,
                                    n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_checkpoint,
                                    early_stopping=early_stopping, verbose=verbose)
        elif model_type=="ffn":
            train_methods.train_ffn(x_train_path, y_train_path, x_val_path, y_val_path,
                  input_size, output_size, model_name, hidden_activation,
                  out_activation, hidden_dims, layers, kernel_initializer, kernel_regularizer, dropouts,
                  n_epochs, batch_size, learning_rate,
                  save_checkpoint,
                  early_stopping, verbose)
