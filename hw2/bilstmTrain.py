import sys
import argparse

from train_script import train

if __name__ == '__main__':

    # create the parser
    parser = argparse.ArgumentParser(description='train bilstm model')

    parser.add_argument("repr", type=str ,choices=['a', 'b', 'c', 'd'], help='which type of word embeddings representation to use')
    parser.add_argument("train_path", type=str, help="a path to training file")
    parser.add_argument("model_path", type=str, help="a path to test file")
    parser.add_argument("model_config_path", type=str,
                                 help="path to a json file containing model hyper parameters")
    parser.add_argument("training_config_path", type=str,
                                 help="path to a json file containing hyper parameters for training procedure")
    parser.add_argument("--dev_path", type=str, required=False,  help="a path to dev file")
    parser.add_argument('--pos_or_ner', type=str, required=False, help="whether to calculate accuracy the ner or normal way", default="pos")

    model_type_convertor = {'a' : "lstm", 'c' : "lstm_sub_words",
                        'b' : "lstm_char_embeddings", 'd': "lstm_char_word_embeddings"}
    # parse the args
    args = parser.parse_args(sys.argv[1:])
    repr = args.repr
    train_path = args.train_path
    run_name = args.model_path
    if args.dev_path:
        test_path = args.dev_path
    else:
        test_path = args.train_path
    pos_or_ner = args.pos_or_ner
    model_config_path = args.model_config_path
    training_config_path = args.training_config_path
    model_type = model_type_convertor[repr] + '_' + pos_or_ner

    train(run_name, model_type, train_path, test_path, model_config_path, training_config_path)
