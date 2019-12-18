import torch
from torch.utils import data
from factory_classes import ModelsFactory, MappersFactory, ConfigsFactory, PredictorsFactory, DatasetsFactory
from models import BaseModel
from mappers import BaseMapper
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
    model_config_params = config_data["state"]

    mapper_data = checkpoint_data["mapper"]
    mapper_name = mapper_data["name"]
    mapper_state = mapper_data["state"]

    # load a config
    model_name_case_insensitive = model_name.lower()
    model_config: BaseConfig = configs_factory(model_name_case_insensitive)
    model_config.from_dict(model_config_params)

    # create a mapper
    trained_mapper: BaseMapper = mappers_factory.get_from_mapper_name(mapper_name)
    trained_mapper.deserialize(mapper_state)

    # create a model
    trained_model: BaseModel = models_factory.get_from_model_name(model_name, model_config, trained_mapper)
    trained_model.load_state_dict(model_state)

    return trained_model, model_name


def save_predictions_to_file(test_path: str, predictions: list, save_model_path: str):
    index = 0
    with open(save_model_path, "w") as out_file:
        with open(test_path, "r", encoding="utf8") as test_path:
            for line in test_path:

                # skip empty line (end of sentence)
                if line == "\n":
                    out_file.write("\n")  # end of sentence in prediction file
                else:  # valid line of a word we need to label
                    word = line[:-1]
                    label = predictions[index]
                    prediction_line = f"{word} {label}\n"
                    out_file.write(prediction_line)
                    index += 1  # don't forge to move to next prediction


def inference(test_path: str, inference_config_path: str, saved_model_path: str, save_predictions_path: str) -> None:
    # initiate factory object
    config_factory = ConfigsFactory()
    predictors_factory = PredictorsFactory()
    dataset_factory = DatasetsFactory()

    # load trained model class and initiate a predictor
    inference_config = config_factory("inference").from_json_file(inference_config_path)
    model, model_name = load_trained_model(saved_model_path)
    model: BaseModel
    mapper = model.mapper
    model_type = inference_config["model_type"]
    predictor = predictors_factory(inference_config, mapper, model_type)

    # check if model is a model with sub word units
    if "window" in model_type:
        if "SubWords" in model_name:
            dataset_name = "WindowWithSubWordsDataset"
        else:
            dataset_name = "WindowDataset"
    else:
        dataset_name = "RegularLanguageDataset"

    # create dataset object and preform inference
    test_dataset = dataset_factory.get_from_dataset_name(dataset_name, test_path, mapper)
    test_config_dict = {"batch_size": inference_config["batch_size"], "num_workers": inference_config["num_workers"]}
    test_loader = data.DataLoader(test_dataset, **test_config_dict)

    device = torch.device(inference_config["device"])
    model = model.to(device)
    model.eval()
    predictions = []

    # run the model in batches to create the predictions
    with torch.no_grad():

        for batch_idx, sample in enumerate(test_loader):
            x, _ = sample
            x = x.to(device)
            outputs = model(x)
            batch_predictions = predictor.infer_model_outputs(outputs)

            for prediction in batch_predictions:
                predicted_label = mapper.get_label_from_idx(prediction)
                predictions.append(predicted_label)

    save_predictions_to_file(test_path, predictions, save_predictions_path)

