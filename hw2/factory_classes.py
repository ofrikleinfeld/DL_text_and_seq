import torch.nn as nn
import torch.utils.data as data

from models import BaseModel, WindowTagger, WindowModelWithPreTrainedEmbeddings, WindowModelWithSubWords, AcceptorLSTM
from mappers import BaseMapper, TokenMapper, TokenMapperUnkCategory, TokenMapperWithSubWords, RegularLanguageMapper
from predictors import BasePredictor, WindowModelPredictor, WindowNERTaggerPredictor, AcceptorPredictor
from configs import BaseConfig, ModelConfig, TrainingConfig, WindowTaggerConfig, InferenceConfig, RNNConfig
from datasets import WindowDataset, WindowWithSubWordsDataset, RegularLanguageDataset
from trainers import ModelTrainer, AcceptorTrainer


class ConfigsFactory(object):
    def __call__(self, config_type: str) -> BaseConfig:

        if config_type == "training":
            return TrainingConfig()

        if config_type == "inference":
            return InferenceConfig()

        if "window" in config_type:
            return WindowTaggerConfig()

        if "acceptor" in config_type:
            return RNNConfig()


class MappersFactory(object):
    def __call__(self, parameters_dict: BaseConfig, mapper_name: str) -> BaseMapper:

        # mapper_attributes
        min_frequency = parameters_dict["min_frequency"]
        split_char = parameters_dict["split_char"]

        if "window" in mapper_name:

            # flags
            # check if flags exists
            if "smart_unknown" in parameters_dict:
                return TokenMapperUnkCategory(min_frequency, split_char)

            if "sub_word_units" in parameters_dict:
                return TokenMapperWithSubWords(min_frequency, split_char)

            return TokenMapper(min_frequency, split_char)

        if mapper_name == "acceptor":

            return RegularLanguageMapper(min_frequency, split_char)

    def get_from_mapper_name(self, mapper_name):

        if mapper_name == "TokenMapperWithSubWords":
            return TokenMapperWithSubWords()

        elif mapper_name == "TokenMapperUnkCategory":

            return TokenMapperUnkCategory()

        elif mapper_name == "TokenMapper":
            return TokenMapper()

        elif mapper_name == "RegularLanguageMapper":
            return RegularLanguageMapper()

        else:
            raise AttributeError("Wrong mapper name")


class ModelsFactory(object):

    def __call__(self, parameters_dict: BaseConfig, model_config: ModelConfig, mapper: BaseMapper, model_name: str) -> BaseModel:

        # flags
        if "window" in model_name:
            model_config: WindowTaggerConfig
            sub_words_in_params = "sub_word_units" in parameters_dict
            pre_training_in_params = "pre_trained_embeddings" in parameters_dict

            # both exists
            if sub_words_in_params and pre_training_in_params:
                mapper: TokenMapperWithSubWords
                return WindowModelWithSubWords(model_config, mapper, pre_trained=True,
                                               pre_trained_vocab_path="vocab.txt",
                                               pre_trained_embedding_path="wordVectors.txt")
            # none of them exists
            if not sub_words_in_params and not pre_training_in_params:
                model_config: WindowTaggerConfig
                return WindowTagger(model_config, mapper)

            # just sub words exists
            if sub_words_in_params and not pre_training_in_params:
                mapper: TokenMapperWithSubWords
                return WindowModelWithSubWords(model_config, mapper, pre_trained=False,
                                               pre_trained_vocab_path="vocab.txt",
                                               pre_trained_embedding_path="wordVectors.txt")

            # just pre training exists
            if not sub_words_in_params and pre_training_in_params:
                return WindowModelWithPreTrainedEmbeddings(model_config, mapper,
                                                           pre_trained_vocab_path="vocab.txt",
                                                           pre_trained_embedding_path="wordVectors.txt")

        if model_name == "acceptor":
            model_config: RNNConfig
            mapper: RegularLanguageMapper
            return AcceptorLSTM(model_config, mapper)

    def get_from_model_name(self, model_name, model_config, mapper):

        if model_name == "WindowModelWithSubWords":
            return WindowModelWithSubWords(model_config, mapper)
        elif model_name == "WindowTagger":
            return WindowTagger(model_config, mapper)
        elif model_name == "AcceptorLSTM":
            return AcceptorLSTM(model_config, mapper)


class PredictorsFactory(object):
    def __call__(self, parameters_dict: BaseConfig, mapper, predictor_type: str) -> BasePredictor:

        if predictor_type == "window_ner":
            return WindowNERTaggerPredictor(mapper)
        elif predictor_type == "window_pos":
            return WindowModelPredictor(mapper)
        elif predictor_type == "acceptor":
            return AcceptorPredictor(mapper)


class DatasetsFactory(object):
    def __call__(self, parameters_dict: BaseConfig, file_path: str, mapper: BaseMapper, dataset_type: str) -> data.Dataset:

        if "window" in dataset_type:
            window_size = parameters_dict["window_size"]
            # flags
            if "sub_word_units" in parameters_dict:
                mapper: TokenMapperWithSubWords
                return WindowWithSubWordsDataset(file_path, mapper, window_size)

            else:
                return WindowDataset(file_path, mapper, window_size)

        if dataset_type == "acceptor":
            sequence_length = parameters_dict["sequence_length"]
            return RegularLanguageDataset(file_path, mapper, sequence_length)

    def get_from_dataset_name(self, dataset_name, file_path, mapper, window_size=2, sequence_length=65):

        if dataset_name == "WindowWithSubWordsDataset":
            return WindowWithSubWordsDataset(file_path, mapper, window_size)
        elif dataset_name == "WindowDataset":
            return WindowDataset(file_path, mapper, window_size)
        elif dataset_name == "RegularLanguageDataset":
            return RegularLanguageDataset(file_path, mapper, sequence_length)


class TrainerFactory(object):
    def __call__(self, model: BaseModel, train_config: TrainingConfig,
                 predictor: BasePredictor, loss_function: nn.Module,
                 model_type: str) -> ModelTrainer:

        if "window" in model_type:
            return ModelTrainer(model, train_config, predictor, loss_function)

        if model_type == "acceptor":
            return AcceptorTrainer(model, train_config, predictor, loss_function)


class LossFunctionFactory(object):
    def __call__(self, model_type: str, mapper: BaseMapper = None) -> nn.Module:
        return nn.CrossEntropyLoss()

