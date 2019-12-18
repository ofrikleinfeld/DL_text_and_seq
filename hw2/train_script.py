import torch.nn as nn
from factory_classes import ConfigsFactory, MappersFactory, ModelsFactory, PredictorsFactory, DatasetsFactory, TrainerFactory, LossFunctionFactory


def train(training_unique_name: str, model_type: str, train_path: str, dev_path: str,
          model_config_path: str, training_config_path: str):

    # initiate factory object
    config_factory = ConfigsFactory()
    mappers_factory = MappersFactory()
    models_factory = ModelsFactory()
    predictors_factory = PredictorsFactory()
    datasets_factory = DatasetsFactory()
    trainer_factory = TrainerFactory()
    loss_function_factory = LossFunctionFactory()

    training_config = config_factory("training").from_json_file(training_config_path)
    model_config = config_factory(model_type).from_json_file(model_config_path)
    mapper = mappers_factory(training_config, mapper_name=model_type)
    mapper.create_mapping(train_path)
    train_data = datasets_factory(training_config, train_path, mapper, dataset_type=model_type)
    dev_data = datasets_factory(training_config, dev_path, mapper, dataset_type=model_type)
    model = models_factory(training_config, model_config, mapper,model_name=model_type)
    predictor = predictors_factory(training_config, mapper, predictor_type=model_type)
    ce_loss = loss_function_factory(model_type, mapper)
    trainer = trainer_factory(model, training_config, predictor, ce_loss, model_type=model_type)
    trainer.train(training_unique_name, train_dataset=train_data, dev_dataset=dev_data)
