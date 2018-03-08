import os
import h5py
import abc
from keras.models import Model, Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Input, Embedding, Bidirectional, GRU, Dropout


class TrainResults():
    def __init__(self, train_loss=None, val_loss=None, epoch=None):
        self.val_loss = val_loss
        self.train_loss = train_loss
        self.epoch = epoch


class BaseModel(object):
    def __init__(self, input_size, output_size, model_name="test", hidden_activation="relu", out_activation="sigmoid",
                 kernel_initializer='uniform', kernel_regularizer=None):
        self.model_name = model_name
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.model = None

    def compile(self, metrics=[], optimizer='adam', loss="binary_crossentropy"):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[loss] +metrics)

    def fit(self, x_train, y_train, validation_data=None , n_epochs=10, batch_size=100, callbacks=None, verbose=1):
        return self.model.fit(x_train, y_train, validation_data=validation_data, epochs=n_epochs,
                                 batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    def predict(self, x_test, batch_size=10, verbose=1):
        return self.model.predict(x_test, batch_size=batch_size, verbose=verbose)

    def evaluate(self, x_data, y_data, batch_size=10, verbose=0):
        scores = self.model.evaluate(x_data, y_data, verbose=verbose, batch_size=batch_size)
        return self.model.metrics_names, scores

    def save_weights(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = h5py.File(folder + self.model_name + ".h5", 'w')
        weights = self.model.get_weights()
        for i in range(len(weights)):
            file.create_dataset('weight' + str(i), data=weights[i])
        file.close()

    def load_model_weights(self, filepath):
        # workaround to load weights into new model
        # f = h5py.File(filepath, 'r')
        # weight = []
        # for i in range(len(f.keys())):
        #     weight.append(f['weight' + str(i)][:])
        # self.model.set_weights(weight)
        self.model.load_weights(filepath, by_name=False)

    @abc.abstractmethod
    def build_model(self):
        pass


class FFNModel(BaseModel):
    def __init__(self, input_size, output_size, model_name="ffn",hidden_activation="relu", out_activation="sigmoid",
                 hidden_dims=[32], layers=1, kernel_initializer='uniform', kernel_regularizer=None, dropouts=[0.0]):
        BaseModel.__init__(self,input_size, output_size, model_name, hidden_activation, out_activation,
                           kernel_initializer, kernel_regularizer)
        self.layers = layers
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        # Adding the input layer and the first hidden layer
        input_size=self.input_size
        for l in range(self.layers):
            self.model.add(Dense(units=self.hidden_dims[l], kernel_initializer=self.kernel_initializer, activation=self.hidden_activation,
                             input_dim=input_size))
            self.model.add(Dropout(self.dropouts[l]))
            input_size=self.hidden_dims[l]
        # Adding the second hidden layer
        # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
        # Adding the output layer
        self.model.add(Dense(units=self.output_size, kernel_initializer=self.kernel_initializer, activation=self.out_activation))


class BaseRNNModel(BaseModel):
    def __init__(self, input_size, output_size, max_seq_length, model_name="test", hidden_activation="relu", out_activation="sigmoid",
                 hidden_dim=32, kernel_initializer='uniform', kernel_regularizer=None, recurrent_regularizer=None,
                 input_dropout=0.0, recurrent_dropout=0.0, rnn_unit_type="rnn",  bidirectional=False, embed_dim=32, emb_trainable=True):
        BaseModel.__init__(self, input_size, output_size, model_name, hidden_activation, out_activation,
                           kernel_initializer, kernel_regularizer)
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.emb_trainable = emb_trainable
        self.hidden_dim = hidden_dim
        self.rnn_unit_type = rnn_unit_type
        self.bidirectional = bidirectional
        self.recurrent_regularizer = recurrent_regularizer
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout


class RNNModel(BaseRNNModel):

    def __init__(self, max_seq_length, input_size, output_size, model_name="test", hidden_activation="relu",
                 out_activation="sigmoid", hidden_dim=32, kernel_initializer='uniform', kernel_regularizer=None,
                 recurrent_regularizer=None, input_dropout=0.0, recurrent_dropout=0.0,
                 rnn_unit_type="rnn", bidirectional=False, embed_dim=32, emb_trainable=True):
        BaseRNNModel.__init__(self, input_size, output_size, max_seq_length, model_name, hidden_activation, out_activation,
                              hidden_dim, kernel_initializer, kernel_regularizer, recurrent_regularizer,
                              input_dropout, recurrent_dropout, rnn_unit_type, bidirectional,embed_dim,emb_trainable)
        self.build_model()


    def build_model(self):
        # the model receives sequences of length self.max_seq_length. At each timestep, the vector size is self.features.
        input = Input(shape=(self.max_seq_length,), dtype='int32', name="myinput")
        # the input sequence is encoded into dense vectors of size self.embed_dim.
        # the input value 0 is a special padding value (for sequences with variable length)
        # that should be masked out. Our vocabulary SHOULD start from 1.
        embedded_input = self.embedding_layer()(input)

        # A RNN or a LSTM transforms the input sequences into a single vector (which is the last hidden state of the rnn)
        # This vector has size self.hidden_dim.
        if self.rnn_unit_type == 'rnn':
            recurrent_layer = SimpleRNN(self.hidden_dim, activation=self.hidden_activation,
                                        kernel_initializer=self.kernel_initializer, dropout=self.input_dropout,
                                        recurrent_dropout=self.recurrent_dropout,
                                        recurrent_regularizer=self.recurrent_regularizer, name="rnn")
        elif self.rnn_unit_type == 'lstm':
            recurrent_layer = LSTM(self.hidden_dim, activation=self.hidden_activation,
                                   recurrent_regularizer=self.recurrent_regularizer, dropout=self.input_dropout,
                                   ecurrent_dropout=self.recurrent_dropout,
                                   kernel_initializer=self.kernel_initializer, name="lstm")
        elif self.rnn_unit_type == 'gru':
            recurrent_layer = GRU(self.hidden_dim, activation=self.hidden_activation,
                                   recurrent_regularizer=self.recurrent_regularizer, input_dropout=self.input_dropout,
                                   ecurrent_dropout=self.recurrent_dropout,
                                   kernel_initializer=self.kernel_initializer, name="gru")
        else:
            raise ValueError('Unknown model type')
        # For Bidirectional rnn, the forward and backward states will be concatenated. So the output vector
        # will have size self.hidden_dim*2.
        if self.bidirectional:
            recurrent_output = Bidirectional(recurrent_layer, merge_mode="concat")(embedded_input)
        else: recurrent_output = recurrent_layer(embedded_input)


        # The output layer takes as input the last hidden state of the rnn and applies self.out_activation function.
        # If self.out_activation is softmax (this is what we need), it produces a probability distribution over classes.
        output = Dense(self.output_size, activation=self.out_activation, name="output")(recurrent_output)
        self.model = Model(inputs=input, outputs=output)


    def embedding_layer(self):
        # This layer turns positive integers (indexes) into dense vectors of fixed size.
        # eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        # input_dim is the size of the vocabulary, i.e. the max index that will be turned into a vector.
        # output_dim is the dense vector size.
        return Embedding(input_dim=self.input_size, output_dim=self.embed_dim, mask_zero=True,
                         input_length=self.max_seq_length,
                         trainable=self.emb_trainable, name="embeddings")