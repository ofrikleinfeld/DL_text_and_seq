import torch
from torch.utils import data
from factory_classes import ModelsFactory, MappersFactory, ConfigsFactory, PredictorsFactory, DatasetsFactory
from models import BaseModel
from mappers import BaseMapper
from datasets import WindowDataset
from configs import BaseConfig


def load_trained_model(path_to_pth_file: str):
    checkpoint_data = torch.load(path_to_pth_file)

    # Factories
    models_factory = ModelsFactory()
    configs_factory = ConfigsFactory()
    mappers_factory = MappersFactory()

    # extract serialized_model and create a model object
    model_data = checkpoint_data["model"]
    model_name = model_data["name"]
    model_state = model_data["state"]

    config_data = checkpoint_data["config"]
    model_config_name = config_data["name"]
    model_config_params = config_data["state"]

    mapper_data = checkpoint_data["mapper"]
    mapper_name = mapper_data["name"]
    mapper_state = mapper_data["state"]

    # create a mapper
    trained_mapper: BaseMapper = mappers_factory(mapper_name)
    trained_mapper.deserialize(mapper_state)

    # creat a config object
    model_config: BaseConfig = configs_factory(model_config_name, model_config_params)

    # create a model
    trained_model: BaseModel = models_factory(model_name, model_config, trained_mapper)
    trained_model.load_state_dict(model_state)

    return trained_model


def inference(test_path: str, inference_config_name: str, inference_config_path: str, saved_model_path: str,
              predictor_name: str, dataset_name: str) -> list:
    # initiate factory object
    config_factory = ConfigsFactory()
    predictors_factory = PredictorsFactory()
    dataset_factory = DatasetsFactory()

    # load trained model class and initiate a predictor
    model: BaseModel = load_trained_model(saved_model_path)
    mapper = model.mapper
    predictor = predictors_factory(predictor_name, mapper)

    # create dataset object and preform inference
    test_dataset = dataset_factory(dataset_name, test_path, mapper)
    inference_config = config_factory(inference_config_name).from_json_file(inference_config_path)
    test_config_dict = {"batch_size": inference_config.batch_size, "num_workers": inference_config.num_workers}
    test_loader = data.DataLoader(test_dataset, **test_config_dict)

    device = torch.device(inference_config.device)
    model = model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():

        for batch_idx, sample in enumerate(test_loader):
            x, _ = sample
            x = x.to(device)
            outputs = model(x)
            batch_predictions = predictor.infer_model_outputs(outputs)

            for prediction in batch_predictions:
                predicted_label = mapper.get_label_from_idx(prediction)
                predictions.append(predicted_label)

    return predictions
