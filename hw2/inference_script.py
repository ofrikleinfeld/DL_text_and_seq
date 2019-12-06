from pathlib import Path

import torch
from torch.utils import data

from factory_classes import ModelsFactory, MappersFactory, ConfigsFactory
from models import BaseModel
from mappers import BaseMapper
from datasets import WindowDataset
from configs import BaseConfig, InferenceConfig
from predictors import BasePredictor, WindowModelPredictor


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


if __name__ == '__main__':

    # paths to test dataset config file and saved checkpoints
    test_path = "ner/test"
    inference_config_path = "inference_config.json"
    checkpoints_dir = Path("checkpoints")

    # load trained model class and initiate a predictor
    checkpoint = checkpoints_dir / "NER" / "06-12-19_best_model.pth"
    model: BaseModel = load_trained_model(checkpoint)
    mapper = model.mapper
    predictor = WindowModelPredictor(mapper)

    # create dataset object and preform inference
    test_dataset = WindowDataset(test_path, mapper)

    inference_config = InferenceConfig().from_json_file(inference_config_path)
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
                print(predicted_label)
