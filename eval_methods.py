import model
import utils


def evaluate_rnn(params):
    x_data, y_data = utils.load_data(params["eval_x_data"], params["eval_y_data"])
    rnn_model = model.RNNModel(max_seq_length=params["max_seq_length"], input_size=params["input_size"],
                               output_size=params["output_size"], embed_dim=params["embed_dim"],
                               emb_trainable=params["emb_trainable"], model_name=params["model_name"],
                               hidden_activation=params["hidden_activation"], out_activation=params["out_activation"],
                               hidden_dim=params["hidden_dims"][0], kernel_initializer=params["kernel_initializer"],
                               kernel_regularizer=params["kernel_regularizer"], recurrent_regularizer=params["recurrent_regularizer"],
                               input_dropout=0.0, recurrent_dropout=0.0,
                               rnn_unit_type=params["rnn_unit_type"],  bidirectional=params["bidirectional"])
    utils.load_model(params["eval_weights_ckpt"], rnn_model, params["learning_rate"])
    print("Model from checkpoint %s was loaded." % params["eval_weights_ckpt"])
    if y_data is not None:
        metrics_names, scores = rnn_model.evaluate(x_data, y_data, batch_size=params["batch_size"], verbose=params["verbose"])
        loss = scores[0]
        print("Evaluation loss: %.3f"%loss)
    predictions = rnn_model.predict(x_data, batch_size=params["batch_size"], verbose=params["verbose"])
    utils.save_predictions(predictions, params["eval_res_folder"], rnn_model.model_name+"_predictions.txt")


def evaluate_ffn(params):
    x_data, y_data = utils.load_data(params["eval_x_data"], params["eval_y_data"])
    fnn_model = model.FFNModel(input_size=params["input_size"], output_size=params["output_size"],
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