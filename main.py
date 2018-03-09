import os
import train_methods
import eval_methods
import argparse
import cPickle

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Data
    parser.add_argument("--x_train_path", type=str, default=None, help="Input train path.")
    parser.add_argument("--y_train_path", type=str, default=None, help="Output train path.")
    parser.add_argument("--x_val_path", type=str, default=None, help="Input validation path.")
    parser.add_argument("--y_val_path", type=str, default=None, help="Output validation path.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder to save the trained models.")

    # Model
    parser.add_argument("--model_type", type=str, default="ffn", help="ffn|rnn. Type of model to run.")
    parser.add_argument("--model_name", type=str, default="test_model", help="Name of the model.")
    parser.add_argument("--input_size", type=int, default=100, help="Size of the input features.")
    parser.add_argument("--output_size", type=int, default=1, help="Output size.")
    parser.add_argument("--max_seq_length", type=int, default=100, help="Maximum sequence length.")

    parser.add_argument("--hidden_activation", type=str, default="relu", help="Hidden layer activation function.")
    parser.add_argument("--out_activation", type=str, default="sigmoid", help="Output layer activation function.")
    parser.add_argument("--kernel_initializer", type=str, default="glorot_uniform", help="Kernels initializer.")
    parser.add_argument("--rnn_unit_type", type=str, default="rnn", help="rnn | gru | lstm. Type of RNN hidden unit.")
    parser.add_argument("--hidden_dims", type=str, default="32",
                        help="A comma separated list of hidden sizes for each of the model layers. For RNN, only 1 layer"
                             " is supported.")
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding layer size.")
    parser.add_argument("--emb_trainable", type="bool", default=True, help="Whether to use a trainable embeddings layer.")
    parser.add_argument("--bidirectional", type="bool", default=False, help="Whether to use o bidirectional RNN.")
    parser.add_argument("--kernel_regularizer", type=str, default=None, help="Kernel regularizer for FFN model.")
    parser.add_argument("--recurrent_regularizer", type=str, default=None, help="Recurrent layer regularizer.")
    parser.add_argument("--recurrent_dropout", type=float, default=0.0, help="Dropout for hidden to hidden units.")
    parser.add_argument("--input_dropout", type=float, default=0.0, help="Dropout for input to hidden units.")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers. For RNN, only 1 layer is supported.")
    parser.add_argument("--dropouts", type=str, default="0.0", help="A comma separated list of dropout values for each of "
                                                                  "the FFN layers.")

    # Evaluation
    parser.add_argument("--eval_weights_ckpt", type=str, default=None, help="Checkpoint to load model weights for evaluation.")
    parser.add_argument("--eval_x_data", type=str, default=None, help="Input evaluation path.")
    parser.add_argument("--eval_y_data", type=str, default=None, help="Output evaluation path.")
    parser.add_argument("--eval_res_folder", type=str, default=None, help="Output folder to save evaluation results.")


    # Training
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochss.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--save_checkpoint", type="bool", default=True, help="Whether to save a checkpoint at each epoch.")
    parser.add_argument("--early_stopping", type="bool", default=True, help="Whether to do early stopping during training.")
    parser.add_argument("--verbose", type=int, default=1, help="0|1. Verbose for training/evaluation.")
    parser.add_argument("--loss", type=str, default="binary_crossentropy", help="Loss function.")

    # Other
    parser.add_argument("--params_file", type=str, default=None, help="Load parameters from file.")
    parser.add_argument("--gpu", type=str, default=None, help="Which GPU to use.")


def process_params(params):
    params["hidden_dims"] = [int(token) for token in params["hidden_dims"].split(",")]
    params["dropouts"] = [float(token) for token in params["dropouts"].split(",")]
    # Add error messages for inconsistent parameter combinations.
    if params["layers"]>1 and params["model_type"]=="rnn":
        raise ValueError("We only support RNN with 1 layer.")


def save_params(params):
    print("Params saved in %s."%os.path.join(params["out_dir"],params["model_name"]+"_params.pickle"))
    cPickle.dump(params, open(os.path.join(params["out_dir"],params["model_name"]+"_params.pickle"),"wb"))


def load_params(filepath):
    return cPickle.load(open(filepath))


def ensure_compatible_params(loaded_params, input_params):
    keys = input_params.keys()
    # if there are new keys in input params, add them to the loaded params.
    for key in keys:
        if key not in loaded_params:
            loaded_params[key]=input_params[key]
    # update the values of the following keys
    keys_to_update = {"eval_weights_ckpt", "eval_x_data", "eval_y_data", "eval_res_folder", "verbose"}
    for key in keys:
        if key in keys_to_update and loaded_params[key] != input_params[key]:
            print("# Updating hparams.%s: %s -> %s" %
                            (key, loaded_params[key],
                             input_params[key]))
            loaded_params[key] = input_params[key]

    return loaded_params


def run_main(params):
    import os
    if params.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    params = vars(params)
    if params["params_file"] is not None:
        loaded_params = load_params(params["params_file"])
        params = ensure_compatible_params(loaded_params, params)
    else:
        process_params(params)
        save_params(params)

    model_type=params["model_type"]

    if params["eval_weights_ckpt"] is not None:
        if model_type == "rnn":
            eval_methods.evaluate_rnn(params)
        elif model_type == "ffn":
            eval_methods.evaluate_ffn(params)
    else:
        if model_type=="rnn":
            train_methods.train_rnn(params)
        elif model_type=="ffn":
            train_methods.train_ffn(params)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # add the possible command line arguments.
    add_arguments(parser)
    params, unparsed = parser.parse_known_args()
    run_main(params)

