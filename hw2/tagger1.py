import argparse
import sys
from train_script import train
from inference_script import inference

if __name__ == '__main__':

    # training and inference setting and parameters
    parser = argparse.ArgumentParser(description='NER/POS models training and prediction application')
    subparsers = parser.add_subparsers()
    # parser.add_argument("--mode", type=str, required=True, choices=["training", "inference"])

    # create the parser for the "training" command
    training_parser = subparsers.add_parser('training')
    training_parser.add_argument("--name", type=str, required=True, metavar='ner_tagger',
                                 help='unique name of the training procedure (used for checkpoint saving')
    training_parser.add_argument("--train_path", type=str, required=True,
                                 help="a path to training file")
    training_parser.add_argument("--dev_path", type=str, required=True,
                                 help="a path to a development set file")
    training_parser.add_argument("--model_config_path", type=str, required=False, default="window_tagger_config.json",
                                 help="path to a json file containing model hyper parameters")
    training_parser.add_argument("--training_config_path", type=str, required=False, default="training_config.json",
                                 help="path to a json file containing hyper parameters for training procedure")
    training_parser.add_argument("--unknown_token", type=str, required=True, choices=["unknown", "smart_unknown"],
                                 default="smart_unknown", help="a choice how to deal with tokens not seen on train data")
    training_parser.add_argument("--model_name", type=str, required=False, default="WindowTagger",
                                 help="which model to use during training")
    training_parser.add_argument("--model_type", type=str, required=True, choices=["pos", "ner"],
                                 help="which model to use during training (effects accuracy computation)")

    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument("--test_path", type=str, required=True,
                                  help="a path to a test set file")
    inference_parser.add_argument("--trained_model_path", type=str, required=True,
                                  help="a path to a trained model checkpoint")
    inference_parser.add_argument("--inference_config_path", type=str, required=False, default="inference_config.json",
                                  help="path to a json file containing model hyper parameters for inference procedure")
    inference_parser.add_argument("--model_type", type=str, required=True, choices=["pos", "ner"], default="pos",
                                  help="which predictor to use during training (effects accuracy computation)")
    args = parser.parse_args(sys.argv[1:])
    mode = sys.argv[1]

    if mode == "training":
        name = args.name
        train_path = args.train_path
        dev_path = args.dev_path
        model_config_path = args.model_config_path
        training_config_path = args.training_config_path
        train_config_name = "TrainingConfig"
        model_config_name = "WindowTaggerConfig"
        model_name = args.model_name
        model_type = args.model_type

        if args.unknown_token == "unknown":
            mapper_name = "TokenMapper",
        elif args.unknown_token == "smart_unknown":
            mapper_name = "TokenMapperUnkCategory"
        else:
            raise AttributeError("Wrong mapper name")

        if args.model_type == "pos":
            predictor_name = "WindowModelPredictor"
        elif args.model_type == "ner":
            predictor_name = "WindowNERTaggerPredictor"
        else:
            raise AttributeError("Wrong predictor name")

        # train the model
        train(name, model_type, train_path, dev_path, model_config_name, model_config_path, train_config_name,
              training_config_path, model_name, mapper_name, predictor_name)

    else:  # mode is inference
        test_path = args.test_path
        trained_model_path = args.trained_model_path
        inference_config_path = args.inference_config_path
        inference_config_name = "InferenceConfig"

        if args.model_type == "pos":
            predictor_name = "WindowModelPredictor"
        elif args.predictor_name == "ner":
            predictor_name = "WindowNERTaggerPredictor"
        else:
            raise AttributeError("Wrong predictor name")

        predictions = inference(test_path, inference_config_name, inference_config_path, trained_model_path, predictor_name)
        print(len(predictions))
        print(predictions[:100])
