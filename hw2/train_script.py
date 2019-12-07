import torch.nn as nn
from factory_classes import ConfigsFactory, MappersFactory, ModelsFactory, PredictorsFactory, DatasetsFactory
from trainers import ModelTrainer


def train(training_unique_name: str, train_path: str, dev_path: str,
          model_config_path: str, training_config_path: str):

    # initiate factory object
    config_factory = ConfigsFactory()
    mappers_factory = MappersFactory()
    models_factory = ModelsFactory()
    predictors_factory = PredictorsFactory()
    datasets_factory = DatasetsFactory()

    # create config object
    model_config = config_factory("model").from_json_file(model_config_path)
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
