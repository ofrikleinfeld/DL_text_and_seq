import torch.nn as nn
from factory_classes import ConfigsFactory, MappersFactory, ModelsFactory, PredictorsFactory, DatasetsFactory
from trainers import ModelTrainer


def train(training_unique_name: str, model_type: str, train_path: str, dev_path: str,
          model_config_name: str, model_config_path: str,
          training_config_name: str, training_config_path: str,
          model_name: str, mapper_name: str, predictor_name: str,
          dataset_name: str):

    # initiate factory object
    config_factory = ConfigsFactory()
    mappers_factory = MappersFactory()
    models_factory = ModelsFactory()
    predictors_factory = PredictorsFactory()
    datasets_factory = DatasetsFactory()

    # create config object
    model_config = config_factory(model_config_name).from_json_file(model_config_path)
    training_config = config_factory(training_config_name).from_json_file(training_config_path)

    # read training file and create a mapping from token to indices
    min_frequency = 5
    if model_type == "pos":
        split_char = " "
    else:
        split_char = "\t"
    mapper = mappers_factory(mapper_name, min_frequency, split_char)
    mapper.create_mapping(train_path)

    # create dataset for training and dev
    train_data = datasets_factory(dataset_name, train_path, mapper)
    dev_data = datasets_factory(dataset_name, dev_path, mapper)

    # create a model
    model = models_factory(model_name, model_config, mapper)

    # create a trainer object, predictor, and a loss function
    # then start training
    ce_loss = nn.CrossEntropyLoss()
    predictor = predictors_factory(predictor_name, mapper)
    trainer = ModelTrainer(model, training_config, predictor, ce_loss)
    trainer.train(training_unique_name, train_dataset=train_data, dev_dataset=dev_data)
