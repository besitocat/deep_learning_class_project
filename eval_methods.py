import model
import utils


def evaluate_rnn(ckpt_weights_file, eval_res_folder, x_data_path, y_data_path, max_seq_length, input_size, output_size, model_name,
                       hidden_activation, out_activation, hidden_dim, kernel_initializer, kernel_regularizer,
                       recurrent_regularizer, rnn_unit_type, bidirectional, embed_dim, emb_trainable, learning_rate,
                       batch_size, verbose):
    x_data, y_data = utils.create_seq_input_data(x_data_path, y_data_path, max_seq_length)
    rnn_model = model.RNNModel(max_seq_length=max_seq_length, input_size=input_size, output_size=output_size,
                               embed_dim=embed_dim, emb_trainable=emb_trainable,
                               model_name=model_name, hidden_activation=hidden_activation,
                               out_activation=out_activation, hidden_dim=hidden_dim,
                               kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                               recurrent_regularizer=recurrent_regularizer, input_dropout=0.0,
                               recurrent_dropout=0.0, rnn_unit_type=rnn_unit_type,
                               bidirectional=bidirectional)
    utils.load_model(ckpt_weights_file, rnn_model, learning_rate)
    print("Model from checkpoint %s was loaded." % ckpt_weights_file)
    if y_data is not None:
        metrics_names, scores = rnn_model.evaluate(x_data, y_data, batch_size=batch_size, verbose=verbose)
        loss = scores[0]
        print("Evaluation loss: %.3f"%loss)
    predictions = rnn_model.predict(x_data, batch_size=batch_size, verbose=verbose)
    utils.save_predictions(eval_res_folder, predictions)


def evaluate_ffn(ckpt_weights_file, eval_res_folder, x_data_path, y_data_path, input_size, output_size, model_name, hidden_activation,
                       out_activation, hidden_dims, layers, kernel_initializer, kernel_regularizer, learning_rate,
                       batch_size, verbose):
    x_data, y_data = utils.create_nonseq_input_data(x_data_path, y_data_path)
    fnn_model = model.FFNModel(input_size=input_size, output_size=output_size, model_name=model_name,
                               hidden_activation=hidden_activation, out_activation=out_activation,
                               hidden_dims=hidden_dims,
                               layers=layers, kernel_initializer=kernel_initializer,
                               kernel_regularizer=kernel_regularizer,
                               dropouts=[0.0])
    utils.load_model(ckpt_weights_file, fnn_model, learning_rate)
    if y_data is not None:
        print("Model from checkpoint %s was loaded." % ckpt_weights_file)
        metrics_names, scores = fnn_model.evaluate(x_data, y_data, batch_size=batch_size, verbose=verbose)
        loss = scores[0]
        print("Evaluation loss: %.3f" % loss)
    predictions = fnn_model.predict(x_data, batch_size=batch_size, verbose=verbose)
    utils.save_predictions(eval_res_folder, predictions)