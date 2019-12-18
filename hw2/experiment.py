import argparse
import sys

import torch.nn as nn

from factory_classes import ConfigsFactory, MappersFactory, ModelsFactory, PredictorsFactory, DatasetsFactory
from trainers import AcceptorTrainer


def train(training_unique_name: str, train_path: str, dev_path: str,
          model_config_path: str, training_config_path: str):

    # initiate factory object
    config_factory = ConfigsFactory()
    mappers_factory = MappersFactory()
    models_factory = ModelsFactory()
    predictors_factory = PredictorsFactory()
    datasets_factory = DatasetsFactory()

    # create config object
    model_config = config_factory("acceptor").from_json_file(model_config_path)
    training_config = config_factory("training").from_json_file(training_config_path)

    # create a mapper
    mapper = mappers_factory(training_config)
    mapper.create_mapping(train_path)

    # create dataset for training and dev
    train_data = datasets_factory(training_config, train_path, mapper)
    dev_data = datasets_factory(training_config, dev_path, mapper)

    # create a model
    model = models_factory(training_config, model_config, mapper)

    # create a trainer object, predictor, and a loss function
    # then start training
    ce_loss = nn.CrossEntropyLoss()
    predictor = predictors_factory(training_config, mapper)
    trainer = ModelTrainer(model, training_config, predictor, ce_loss)
    trainer.train(training_unique_name, train_dataset=train_data, dev_dataset=dev_data)




if __name__ == '__main__':

    # training and inference setting and parameters
    parser = argparse.ArgumentParser(description='Run an Acceptor RNN network over a dataset')
    subparsers = parser.add_subparsers()

    # create the parser for the "training" command
    training_parser = subparsers.add_parser('training')
    training_parser.add_argument("--name", type=str, required=True, metavar='regular_language',
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
    inference_parser.add_argument("--test_path", type=str, required=True,
                                  help="a path to a test set file")
    inference_parser.add_argument("--trained_model_path", type=str, required=True,
                                  help="a path to a trained model checkpoint (.pth file")
    inference_parser.add_argument("--save_output_path", type=str, required=True,
                                  help="a path to a save the prediction results ")
    inference_parser.add_argument("--inference_config_path", type=str, required=True,
                                  help="path to a json file containing model hyper parameters for inference procedure")