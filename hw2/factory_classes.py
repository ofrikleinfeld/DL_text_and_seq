import torch.utils.data as data
from models import BaseModel, WindowTagger, WindowModelWithPreTrainedEmbeddings, WindowModelWithSubWords
from mappers import BaseMapper, TokenMapper, TokenMapperUnkCategory, TokenMapperWithSubWords
from predictors import BasePredictor, WindowModelPredictor, WindowNERTaggerPredictor
from configs import BaseConfig, TrainingConfig, WindowTaggerConfig, InferenceConfig
from datasets import WindowDataset, WindowWithSubWordsDataset


class ConfigsFactory(object):
    def __call__(self, class_name: str, *constructor_attributes) -> BaseConfig:

        if class_name == "TrainingConfig":
            return TrainingConfig(*constructor_attributes)

        if class_name == "WindowTaggerConfig":
            return WindowTaggerConfig(*constructor_attributes)

        if class_name == "InferenceConfig":
            return InferenceConfig(*constructor_attributes)

        raise AttributeError("Unknown config name")


class MappersFactory(object):
    def __call__(self, class_name: str, *constructor_attributes) -> BaseMapper:

        if class_name == "TokenMapper":
            return TokenMapper(*constructor_attributes)

        if class_name == "TokenMapperUnkCategory":
            return TokenMapperUnkCategory(*constructor_attributes)

        if class_name == "TokenMapperWithSubWords":
            return TokenMapperWithSubWords(*constructor_attributes)

        raise AttributeError("Unknown mapper name")


class ModelsFactory(object):
    def __call__(self, class_name: str, *constructor_attributes) -> BaseModel:

        if class_name == "WindowTagger":
            return WindowTagger(*constructor_attributes)

        if class_name == "WindowModelWithPreTrainedEmbeddings":
            pre_trained_attributes = ("vocab.txt", "wordVectors.txt")
            constructor_attributes += pre_trained_attributes
            return WindowModelWithPreTrainedEmbeddings(*constructor_attributes)

        if class_name == "WindowModelWithSubWords":
            return WindowModelWithSubWords(*constructor_attributes)

        raise AttributeError("Unknown model name")


class PredictorsFactory(object):
    def __call__(self, class_name: str, *constructor_attributes) -> BasePredictor:

        if class_name == "WindowModelPredictor":
            return WindowModelPredictor(*constructor_attributes)

        if class_name == "WindowNERTaggerPredictor":
            return WindowNERTaggerPredictor(*constructor_attributes)

        raise AttributeError("Unknown predictor name")


class DatasetsFactory(object):
    def __call__(self, class_name: str, *constructor_attributes) -> data.Dataset:

        if class_name == "WindowDataset":
            return WindowDataset(*constructor_attributes)

        if class_name == "WindowWithSubWordsDataset":
            return WindowWithSubWordsDataset(*constructor_attributes)

        raise AttributeError("Unknown dataset name")
