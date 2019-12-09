import torch.utils.data as data
from models import BaseModel, WindowTagger, WindowModelWithPreTrainedEmbeddings, WindowModelWithSubWords
from mappers import BaseMapper, TokenMapper, TokenMapperUnkCategory, TokenMapperWithSubWords
from predictors import BasePredictor, WindowModelPredictor, WindowNERTaggerPredictor
from configs import BaseConfig, TrainingConfig, WindowTaggerConfig, InferenceConfig
from datasets import WindowDataset, WindowWithSubWordsDataset


class ConfigsFactory(object):
    def __call__(self, config_type: str) -> BaseConfig:

        if config_type == "training":
            return TrainingConfig()

        if config_type == "model":
            return WindowTaggerConfig()

        if config_type == "inference":
            return InferenceConfig()


class MappersFactory(object):
    def __call__(self, parameters_dict: BaseConfig) -> BaseMapper:

        # mapper_attributes
        min_frequency = parameters_dict["min_frequency"]
        split_char = parameters_dict["split_char"]

        # flags
        smart_unknown = parameters_dict["smart_unknown"]
        sub_word_units = parameters_dict["sub_word_units"]

        if sub_word_units:
            return TokenMapperWithSubWords(min_frequency, split_char)

        if smart_unknown:
            return TokenMapperUnkCategory(min_frequency, split_char)

        return TokenMapper(min_frequency, split_char)

    def get_from_mapper_name(self, mapper_name):

        if mapper_name == "TokenMapperWithSubWords":
            return TokenMapperWithSubWords()
        elif mapper_name == "TokenMapperUnkCategory":
            return TokenMapperUnkCategory()
        elif mapper_name == "TokenMapper":
            return TokenMapper()
        else:
            raise AttributeError("Wrong mapper name")


class ModelsFactory(object):

    def __call__(self, parameters_dict: BaseConfig, model_config, mapper) -> BaseModel:

        # flags
        sub_word_units = parameters_dict["sub_word_units"]
        pre_trained_embeddings = parameters_dict["pre_trained_embeddings"]

        if sub_word_units:
            if pre_trained_embeddings:
                return WindowModelWithSubWords(model_config, mapper, pre_trained=True,
                                               pre_trained_vocab_path="vocab.txt", pre_trained_embedding_path="wordVectors.txt")
            else:
                return WindowModelWithSubWords(model_config, mapper, pre_trained=False)

        if pre_trained_embeddings:
            return WindowModelWithPreTrainedEmbeddings(model_config, mapper, pre_trained_vocab_path="vocab.txt",
                                                       pre_trained_embedding_path="wordVectors.txt")
        return WindowTagger(model_config, mapper)

    def get_from_model_name(self, model_name, model_config, mapper):

        if model_name == "WindowModelWithSubWords":
            return WindowModelWithSubWords(model_config, mapper)
        else:
            return WindowTagger(model_config, mapper)


class PredictorsFactory(object):
    def __call__(self, parameters_dict: BaseConfig, mapper) -> BasePredictor:

        model_type = parameters_dict["model_type"].lower()
        if model_type == "ner":
            return WindowNERTaggerPredictor(mapper)

        return WindowModelPredictor(mapper)


class DatasetsFactory(object):
    def __call__(self, parameters_dict: BaseConfig, file_path: str, mapper, window_size: int = 2) -> data.Dataset:

        # flags
        sub_word_units = parameters_dict["sub_word_units"]

        if sub_word_units:
            return WindowWithSubWordsDataset(file_path, mapper, window_size)

        else:
            return WindowDataset(file_path, mapper, window_size)

    def get_from_dataset_name(self, dataset_name, file_path, mapper, window_size=2):

        if dataset_name == "WindowWithSubWordsDataset":
            return WindowWithSubWordsDataset(file_path, mapper, window_size)
        else:
            return WindowDataset(file_path, mapper, window_size)