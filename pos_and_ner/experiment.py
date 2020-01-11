import sys
import argparse

from pos_and_ner.train_script import train

if __name__ == '__main__':

    # create the parser
    parser = argparse.ArgumentParser(description='train and evaluate an acceptor model on train and test sets')

    parser.add_argument("--name", type=str, required=True, metavar='regex', help='unique name of the training procedure (used for checkpoint saving')
    parser.add_argument("--train_path", type=str, required=True,  help="a path to training file")
    parser.add_argument("--test_path", type=str, required=True,  help="a path to test file")
    parser.add_argument("--model_config_path", type=str, required=True, help="path to a json file containing model hyper parameters")
    parser.add_argument("--training_config_path", type=str, required=True, help="path to a json file containing hyper parameters for training procedure")

    # parse the args
    args = parser.parse_args(sys.argv[1:])
    run_name = args.name
    train_path = args.train_path
    test_path = args.test_path
    model_config_path = args.model_config_path
    training_config_path = args.training_config_path
    model_type = "acceptor"

    train(run_name, model_type, train_path, test_path, model_config_path, training_config_path)
