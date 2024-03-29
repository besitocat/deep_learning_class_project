import model
import utils
from keras import backend as K
import numpy as np
import os
import cPickle


def get_attention_weights(model, data):
    inp = model.model.input  # input placeholder
    functor = K.function([inp] + [K.learning_phase()], [model.attention_scores])
    # Testing
    attention_weights = functor([data, 1.])
    return attention_weights[0]


def evaluate_rnn(params):
    x_data, y_data = utils.load_data(params["eval_x_data"], params["eval_y_data"])
    rnn_model = model.RNNModel(max_seq_length=x_data.shape[1], input_size=params["input_size"],
                               output_size=params["output_size"], embed_dim=params["embed_dim"],
                               emb_trainable=params["emb_trainable"], model_name=params["model_name"],
                               hidden_activation=params["hidden_activation"], out_activation=params["out_activation"],
                               hidden_dim=params["hidden_dims"][0], kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"], recurrent_regularizer=params["recurrent_regularizer"],
                               input_dropout=0.0, recurrent_dropout=0.0,
                               rnn_unit_type=params["rnn_unit_type"],  bidirectional=params["bidirectional"],
                               attention=params["attention"], embs_matrix=params["embs_matrix"])
    utils.load_model(params["eval_weights_ckpt"], rnn_model, params["learning_rate"])
    print("Model from checkpoint %s was loaded." % params["eval_weights_ckpt"])
    # if y_data is not None:
    #     metrics_names, scores = rnn_model.evaluate(x_data, y_data, batch_size=params["batch_size"], verbose=params["verbose"])
    #     loss = scores[0]
    #     print("Evaluation loss: %.3f"%loss)
    sample_idxs = np.random.randint(x_data.shape[0], size=params["attn_sample_size"])
    x_data_sample = x_data[sample_idxs, :]
    cPickle.dump(sample_idxs, open(os.path.join(params["eval_res_folder"], "sample_idxs.pickle"),"wb"))
    if params["attention"]:
        attention_weights=get_attention_weights(rnn_model, x_data_sample)
        print("Attention weights shape: ",attention_weights.shape)
        cPickle.dump(attention_weights, open(os.path.join(params["eval_res_folder"],"attn_weights.pickle"),"wb"))
        vocab = cPickle.load(open(params["vocab_file"]))
        if params["plot_attn"]:
            inverse_vocab = {value:key for key,value in vocab.items()}
            utils.plot_attention_weights(x_data_sample,  attention_weights, inverse_vocab,  params["eval_res_folder"],ids=sample_idxs)

    predictions = rnn_model.predict(x_data, batch_size=params["batch_size"], verbose=params["verbose"])
    utils.save_predictions(predictions, params["eval_res_folder"], rnn_model.model_name+"_predictions.txt")


def evaluate_ffn(params):
    x_data, y_data = utils.load_data(params["eval_x_data"], params["eval_y_data"])
    fnn_model = model.FFNModel(input_size=x_data.shape[1], output_size=params["output_size"],
                               model_name=params["model_name"],
                               hidden_activation=params["hidden_activation"], out_activation=params["out_activation"],
                               hidden_dims=params["hidden_dims"],
                               layers=params["layers"], kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"],
                               dropouts=[0.0])
    utils.load_model(params["eval_weights_ckpt"], fnn_model, params["learning_rate"])
    if y_data is not None:
        print("Model from checkpoint %s was loaded." % params["eval_weights_ckpt"])
        metrics_names, scores = fnn_model.evaluate(x_data, y_data, batch_size=params["batch_size"], verbose=params["verbose"])
        loss = scores[0]
        print("Evaluation loss: %.3f" % loss)
    predictions = fnn_model.predict(x_data, batch_size=params["batch_size"], verbose=params["verbose"])
    utils.save_predictions(predictions, params["eval_res_folder"], fnn_model.model_name + "_predictions.txt")