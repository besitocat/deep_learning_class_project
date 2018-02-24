# deep_learning_class_project - Document Classification

*** Instructions for Running ***:

- Put the original "train-balanced-sarcasm.csv" file in a folder named "sarcasm_data" in the parent folder of this project.

- Run the main.py with pipeline(is_first_run=True, train_with_real_data=True, epochs=10)--> This will generate new files with preprocessed/cleaned comments and everything you need for data transformation + will train a basic NN model and predict on the validation data

If you pass  train_with_real_data=False, you will use the validation file as training file and you have to create a small sample test file and pass it as an argument to pipeline(). For example:

    pipeline(is_first_run=False, train_with_real_data=False,
             epochs=10, sample_test_file=data_prep.root_sarcasm_data_dir + "small_train.csv") 

In the main.py: create_model(input_size) build a NN model. Currently it is a simple Feed Forward network with 1 hidden layer.

Also, we have a naive_bayes_pipeline that you can run to do a simple and quick test that the data are correct.