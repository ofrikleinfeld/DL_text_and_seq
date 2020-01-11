import sys
import argparse

from pos_and_ner.inference_script import inference

if __name__ == '__main__':

    # create the parser
    parser = argparse.ArgumentParser(description='predict using a trained bilstm model')

    parser.add_argument("repr", type=str, choices=['a', 'b', 'c', 'd'], help='which type of word embeddings representation to use')
    parser.add_argument("model_path", type=str, help="a path to a trained model file")
    parser.add_argument("test_path", type=str, help="a path to a test file")
    parser.add_argument('model_type', type=str, choices=["pos", "ner"], help="which model to run POS or NER")
    parser.add_argument("inference_config_path", type=str, help="a path to inference config file")
    parser.add_argument("save_output_path", type=str, help="a path to save the results of the inference on the test set")

    model_type_converter = {'a': "lstm", 'c': "lstm_sub_words", 'b': "lstm_char_embeddings", 'd': "lstm_char_word_embeddings"}

    # parse the args
    args = parser.parse_args(sys.argv[1:])
    repr_ = args.repr
    model_path = args.model_path
    test_path = args.test_path
    model_type = args.model_type
    model_name = model_type_converter[repr_] + '_' + model_type
    inference_config_path = args.inference_config_path
    save_output_path = args.save_output_path
    inference(test_path, inference_config_path, model_path, save_output_path, model_name)
