import sys
import argparse

from inference_script import inference

if __name__ == '__main__':

    # create the parser
    parser = argparse.ArgumentParser(description='train bilstm model')

    parser.add_argument("repr", type=str ,choices=['a', 'b', 'c', 'd'], help='which type of word embeddings representation to use')
    parser.add_argument("model_path", type=str, help="a path to training file")
    parser.add_argument("test_path", type=str, help="a path to test file")
    parser.add_argument("--inference_config_path", required=False, type=str, help="a path to inference config file", default="inference_config.json")
    parser.add_argument("--save_output_path", required=False, type=str, help="a path to save the inference on the test set")
    parser.add_argument('--pos_or_ner', type=str, required=False,
                        help="whether to calculate accuracy the ner or normal way", default="pos")
    model_type_convertor = {'a' : "lstm", 'c' : "lstm_sub_words",
                        'b' : "lstm_char_embeddings", 'd': "lstm_char_word_embeddings"}
    # parse the args
    args = parser.parse_args(sys.argv[1:])
    repr = args.repr
    model_path = args.model_path
    test_path = args.test_path
    pos_or_ner = args.pos_or_ner
    model_type = model_type_convertor[repr] + '_' + pos_or_ner
    inference_config_path = args.inference_config_path
    save_output_path = None
    if not args.save_output_path:
        save_output_path = save_output_path(model_type + '_' + pos_or_ner)
    else:
        save_output_path = args.save_output_path
    inference(test_path, inference_config_path, model_path, save_output_path, model_type)
