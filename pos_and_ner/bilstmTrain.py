import sys
import argparse

from pos_and_ner.train_script import train

if __name__ == '__main__':

    # create the parser
    parser = argparse.ArgumentParser(description='train bilstm model')

    parser.add_argument("repr", type=str, choices=['a', 'b', 'c', 'd'], help='which type of word embeddings representation to use')
    parser.add_argument("train_path", type=str, help="a path to training file")
    parser.add_argument("model_path", type=str, help="a path to save the trained model")
    parser.add_argument("model_config_path", type=str, help="path to a json file containing model hyper parameters")
    parser.add_argument("training_config_path", type=str, help="path to a json file containing hyper parameters for training procedure")
    parser.add_argument('model_type', type=str, choices=["pos", "ner"], help="which model to run POS or NER")
    parser.add_argument("--dev_path", type=str, required=False,  help="a path to dev file")

    model_type_converter = {'a': "lstm", 'c': "lstm_sub_words", 'b': "lstm_char_embeddings", 'd': "lstm_char_word_embeddings"}
    # parse the args
    args = parser.parse_args(sys.argv[1:])
    repr_ = args.repr
    train_path = args.train_path
    run_name = args.model_path
    if args.dev_path:
        test_path = args.dev_path
    else:
        test_path = args.train_path
    model_type = args.model_type
    model_config_path = args.model_config_path
    training_config_path = args.training_config_path
    model_name = model_type_converter[repr_] + '_' + model_type

    train(run_name, model_name, train_path, test_path, model_config_path, training_config_path)
