import train_methods
import eval_methods

#common parameters
out_dir="experiments/trained_models"
output_size = 1
hidden_activation = "relu"
out_activation = "sigmoid"
kernel_initializer = "glorot_uniform"
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
max_seq_length = 150
recurrent_regularizer = None
recurrent_dropout = 0.0
input_dropout = 0.0

#ffn specific parameters
hidden_dims = [32]
layers = 1
dropouts = [0.0]


#rnn input files
model_type="rnn"
model_name = "rnn_test"
x_train_seq_path = "experiments/data/x_train_seq.pickle"
y_train_seq_path = "experiments/data/y_train_seq.pickle"
x_val_seq_path = "experiments/data/x_val_seq.pickle"
y_val_seq_path = "experiments/data/y_val_seq.pickle"
eval_weights_ckpt = "experiments/trained_models/rnn_test.02-0.620.hdf5"
eval_x_data_seq = "experiments/data/x_val_seq.pickle"
eval_y_data_seq = "experiments/data/y_val_seq.pickle"
eval_res_folder = "experiments/results"
eval_res_folder="experiments/results"
input_size=5002
#fnn input files
# model_type="ffn"
# model_name = "ffn_test"
# x_train_noseq_path = "experiments/data/x_train_noseq.pickle"
# y_train_noseq_path = "experiments/data/y_train_noseq.pickle"
# x_val_noseq_path = "experiments/data/x_val_noseq.pickle"
# y_val_noseq_path = "experiments/data/y_val_noseq.pickle"
# eval_weights_ckpt = None
# eval_x_data_noseq = "experiments/data/x_val_noseq.pickle"
# eval_y_data_noseq = "experiments/data/y_val_noseq.pickle"
# input_size=5000
if __name__ == "__main__":

    if eval_weights_ckpt is not None:
        if model_type == "rnn":
            eval_methods.evaluate_rnn(eval_weights_ckpt, eval_res_folder, eval_x_data_seq, eval_y_data_seq, max_seq_length, input_size, output_size, model_name,
                       hidden_activation, out_activation, hidden_dim, kernel_initializer, kernel_regularizer,
                       recurrent_regularizer, rnn_unit_type, bidirectional, embed_dim, emb_trainable, learning_rate,
                       batch_size, verbose)
        elif model_type == "ffn":
            eval_methods.evaluate_ffn(eval_weights_ckpt, eval_res_folder, eval_x_data_noseq, eval_y_data_noseq, input_size, output_size,
                         model_name, hidden_activation,
                         out_activation, hidden_dims, layers, kernel_initializer, kernel_regularizer, learning_rate,
                         batch_size, verbose)
    else:
        if model_type=="rnn":
            train_methods.train_rnn(x_train_path=x_train_seq_path, y_train_path=y_train_seq_path, x_val_path=x_val_seq_path, y_val_path=y_val_seq_path,
                                    out_dir=out_dir, max_seq_length=max_seq_length, input_size=input_size, output_size=output_size,
                                    model_name=model_name, hidden_activation=hidden_activation,
                                    out_activation=out_activation, hidden_dim=hidden_dim, kernel_initializer=kernel_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer, input_dropout=input_dropout, recurrent_dropout=recurrent_dropout,
                                    rnn_unit_type=rnn_unit_type, bidirectional=bidirectional, embed_dim=embed_dim, emb_trainable=emb_trainable,
                                    n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_checkpoint,
                                    early_stopping=early_stopping, verbose=verbose)
        elif model_type=="ffn":
            train_methods.train_ffn(x_train_noseq_path, y_train_noseq_path, x_val_noseq_path, y_val_noseq_path,
                  out_dir, input_size, output_size, model_name, hidden_activation,
                  out_activation, hidden_dims, layers, kernel_initializer, kernel_regularizer, dropouts,
                  n_epochs, batch_size, learning_rate,
                  save_checkpoint,
                  early_stopping, verbose)
