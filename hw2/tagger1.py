import argparse
import sys
from train_script import train
from inference_script import inference

if __name__ == '__main__':

    # training and inference setting and parameters
    parser = argparse.ArgumentParser(description='NER/POS models training and prediction application')
    subparsers = parser.add_subparsers()

    # create the parser for the "training" command
    training_parser = subparsers.add_parser('training')
    training_parser.add_argument("--name", type=str, required=True, metavar='ner_tagger',
                                 help='unique name of the training procedure (used for checkpoint saving')
    training_parser.add_argument("--train_path", type=str, required=True,
                                 help="a path to training file")
    training_parser.add_argument("--dev_path", type=str, required=True,
                                 help="a path to a development set file")
    training_parser.add_argument("--model_config_path", type=str, required=False, default="model_config.json",
                                 help="path to a json file containing model hyper parameters")
    training_parser.add_argument("--training_config_path", type=str, required=False, default="training_config.json",
                                 help="path to a json file containing hyper parameters for training procedure")
    # training_parser.add_argument("--model_type", type=str, required=True, choices=["pos", "ner"],
    #                              help="which model to use during training (effects accuracy computation)")
    # training_parser.add_argument("--smart_unknown", action="store_true", required=False,
    #                              help="Group tokens not seen on train data to distinctive groups such as numbers, dates, captial letters etc.")
    # training_parser.add_argument("--pre_trained_embeddings", action="store_true", required=False,
    #                              help="Whether to use pre trained word embeddings or not")
    # training_parser.add_argument("--sub_word_units", action="store_true", required=False,
    #                              help="Whether to use sub word units embeddings or not")

    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument("--test_path", type=str, required=True,
                                  help="a path to a test set file")
    inference_parser.add_argument("--trained_model_path", type=str, required=True,
                                  help="a path to a trained model checkpoint")
    inference_parser.add_argument("--inference_config_path", type=str, required=False, default="inference_config.json",
                                  help="path to a json file containing model hyper parameters for inference procedure")
    # inference_parser.add_argument("--model_type", type=str, required=True, choices=["pos", "ner"], default="pos",
    #                               help="which predictor to use during training (effects accuracy computation)")
    # inference_parser.add_argument("--sub_word_units", action="store_true", required=False,
    #                               help="Whether to use sub word units embeddings or not")

    args = parser.parse_args(sys.argv[1:])
    mode = sys.argv[1]
    if mode == "training":
        name = args.name
        train_path = args.train_path
        dev_path = args.dev_path
        model_config_path = args.model_config_path
        training_config_path = args.training_config_path
        train(name, train_path, dev_path, model_config_path, training_config_path)

    if mode == "inference":
        test_path = args.test_path
        trained_model_path = args.trained_model_path
        inference_config_path = args.inference_config_path
        # inference_config_name = "InferenceConfig"

        predictions = inference(test_path, inference_config_path, trained_model_path)
        print(len(predictions))
        print(predictions[:100])


    # flags = {
    #     "pre_train": args.pre_trained_embeddings,
    #     "sub_word_units": args.sub_word_units,
    #     "smart_unknown": args.smart_unknown
    # }
    #
    # model_type = args.model_type
    # if model_type == "ner":
    #     predictor_name = "WindowNERTaggerPredictor"
    # else:
    #     predictor_name = "WindowModelPredictor"

    # if mode == "training":
    #     name = args.name
    #     train_path = args.train_path
    #     dev_path = args.dev_path
    #     model_config_path = args.model_config_path
    #     training_config_path = args.training_config_path
    #     train_config_name = "TrainingConfig"
    #     model_config_name = "WindowTaggerConfig"
    #     model_type = args.model_type
    #
    #     # flags
    #     if flags["sub_word_units"]:
    #         dataset_name = "WindowWithSubWordsDataset"
    #         mapper_name = "TokenMapperWithSubWords"
    #         model_name = "WindowModelWithSubWords"
    #         train(name, model_type, train_path, dev_path, model_config_name, model_config_path, train_config_name,
    #               training_config_path, model_name, mapper_name, predictor_name, dataset_name)
    #
    #     else:
    #         if flags["smart_unknown"]:
    #             mapper_name = "TokenMapperUnkCategory"
    #         else:
    #             mapper_name = "TokenMapper"
    #
    #         if flags["pre_train"]:
    #             model_name = "WindowModelWithPreTrainedEmbeddings"
    #         else:
    #             model_name = "WindowTagger"
    #
    #         dataset_name = "WindowDataset"
    #         train(name, model_type, train_path, dev_path, model_config_name, model_config_path, train_config_name,
    #               training_config_path, model_name, mapper_name, predictor_name, dataset_name)
    #
    # else:  # inference mode
    #     test_path = args.test_path
    #     trained_model_path = args.trained_model_path
    #     inference_config_path = args.inference_config_path
    #     inference_config_name = "InferenceConfig"
    #
    #     # flags
    #     if flags["sub_word_units"]:
    #         dataset_name = "WindowWithSubWordsDataset"
    #     else:
    #         dataset_name = "WindowDataset"
    #
    #     predictions = inference(test_path, inference_config_name, inference_config_path,
    #                             trained_model_path, predictor_name, dataset_name)
    #     print(len(predictions))
    #     print(predictions[:100])
