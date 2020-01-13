import argparse
import sys

from pos_and_ner.train_script import train
from pos_and_ner.inference_script import inference

SUPPORTED_MODELS = ["window_ner", "window_pos", "window_pre_trained_ner", "window_pre_trained_pos",
                    "window_sub_words_ner", "window_sub_words_pos", "window_pre_trained_sub_words_ner",
                    "window_pre_trained_sub_words_pos", "acceptor",
                    "lstm_ner", "lstm_pos", "lstm_sub_words_ner", "lstm_sub_words_pos",
                    "lstm_char_embeddings_ner", "lstm_char_embeddings_pos",
                    "lstm_char_word_embeddings_ner", "lstm_char_word_embeddings_pos",
                    "SNLI_attention_vanilla"]

if __name__ == '__main__':

    # training and inference setting and parameters
    parser = argparse.ArgumentParser(description='models training and prediction application')
    subparsers = parser.add_subparsers()

    # create the parser for the "training" command
    training_parser = subparsers.add_parser('training')
    training_parser.add_argument("--name", type=str, required=True, metavar='window_ner',
                                 help='unique name of the training procedure (used for checkpoint saving')
    training_parser.add_argument("--model_type", type=str, required=True, choices=SUPPORTED_MODELS,
                                 help='unique name of the training procedure (used for checkpoint saving')
    training_parser.add_argument("--train_path", type=str, required=True,
                                 help="a path to training file")
    training_parser.add_argument("--dev_path", type=str, required=True,
                                 help="a path to a development set file")
    training_parser.add_argument("--model_config_path", type=str, required=True,
                                 help="path to a json file containing model hyper parameters")
    training_parser.add_argument("--training_config_path", type=str, required=True,
                                 help="path to a json file containing hyper parameters for training procedure")

    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument("--model_type", type=str, required=True, choices=SUPPORTED_MODELS,
                                 help='unique name of the training procedure (used for checkpoint saving')
    inference_parser.add_argument("--test_path", type=str, required=True,
                                  help="a path to a test set file")
    inference_parser.add_argument("--trained_model_path", type=str, required=True,
                                  help="a path to a trained model checkpoint (.pth file")
    inference_parser.add_argument("--save_output_path", type=str, required=True,
                                  help="a path to a save the prediction results ")
    inference_parser.add_argument("--inference_config_path", type=str, required=True,
                                  help="path to a json file containing model hyper parameters for inference procedure")

    args = parser.parse_args(sys.argv[1:])
    mode = sys.argv[1]
    if mode == "training":
        name = args.name
        model_type = args.model_type
        train_path = args.train_path
        dev_path = args.dev_path
        model_config_path = args.model_config_path
        training_config_path = args.training_config_path
        train(name, model_type, train_path, dev_path, model_config_path, training_config_path)

    if mode == "inference":
        test_path = args.test_path
        model_type = args.model_type
        trained_model_path = args.trained_model_path
        inference_config_path = args.inference_config_path
        save_output_path = args.save_output_path

        inference(test_path, inference_config_path, trained_model_path, save_output_path, model_type)
