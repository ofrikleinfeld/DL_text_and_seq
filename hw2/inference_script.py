import torch
from torch.utils import data

from factory_classes import ModelsFactory, MappersFactory, ConfigsFactory, PredictorsFactory, DatasetsFactory
from models import BaseModel
from mappers import BaseMapper, BaseMapperWithPadding
from configs import BaseConfig, ModelConfig
from datasets import BiLSTMDataset


def load_trained_model(path_to_pth_file: str, model_type: str):
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
    mapper_state = mapper_data["state"]

    # load a config
    model_config: ModelConfig = configs_factory(model_type)
    model_config.from_dict(model_config_params)

    # create a mapper
    trained_mapper: BaseMapper = mappers_factory(BaseConfig(), model_type)
    trained_mapper.deserialize(mapper_state)

    # create a model
    trained_model: BaseModel = models_factory(model_name, model_config, trained_mapper, model_type)
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


def inference(test_path: str, inference_config_path: str, saved_model_path: str, save_predictions_path: str, model_type: str) -> None:
    # initiate factory object
    config_factory = ConfigsFactory()
    predictors_factory = PredictorsFactory()
    dataset_factory = DatasetsFactory()

    # load trained model class and initiate a predictor
    inference_config = config_factory("inference").from_json_file(inference_config_path)
    model, _ = load_trained_model(saved_model_path, model_type)
    model: BaseModel
    mapper = model.mapper
    predictor = predictors_factory(inference_config, mapper, model_type)

    # check if model is a model with sub word units

    # create dataset object and preform inference
    test_dataset = dataset_factory(BaseConfig(), test_path, mapper, model_type)
    test_config_dict = {"batch_size": inference_config["batch_size"], "num_workers": inference_config["num_workers"]}
    test_loader = data.DataLoader(test_dataset, **test_config_dict)

    device = torch.device(inference_config["device"])
    model = model.to(device)
    model.eval()
    predictions = []

    # if it an LSTM model we must make sure that the sequence length is the maximum sequence length in the data
    # and to use mask tensor to make sure we only predict for real tokens
    if "lstm" in model_type:
        test_dataset: BiLSTMDataset
        mapper: BaseMapperWithPadding
        padding_index = mapper.get_padding_index()
        max_sequence_length = test_dataset.get_dataset_max_sequence_length()
        test_dataset.sequence_length = max_sequence_length

        # run the model in batches to create the predictions
        with torch.no_grad():

            for batch_idx, sample in enumerate(test_loader):
                x, _ = sample
                x = x.to(device)
                outputs = model(x)

                real_tokens_mask: torch.tensor

                # special case for character embeddings
                if "char_embeddings" in model_type:
                    batch_size, num_words, chars_in_word = x.size()
                    padding_word = torch.tensor([padding_index] * chars_in_word, dtype=torch.uint8).to(device)

                    # fold num words dimension into batch dimension
                    all_batch_words = x.view(batch_size * num_words, chars_in_word)
                    # a valid word is a word that has at least 1 char that is not the char padding
                    real_tokens_mask = ~torch.all(torch.eq(all_batch_words, padding_word), dim=1)

                else:
                    if len(x.size()) == 3:
                        real_tokens_mask = (x[:, 0, :] != padding_index).flatten()
                    else:
                        real_tokens_mask = (x != padding_index).flatten()

                batch_predictions: torch.tensor = predictor.infer_model_outputs(outputs)
                batch_predictions = batch_predictions.flatten()

                for i in range(len(batch_predictions)):
                    if real_tokens_mask[i]:
                        predicted_label = mapper.get_label_from_idx(batch_predictions[i].item())
                        predictions.append(predicted_label)
    else:
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

