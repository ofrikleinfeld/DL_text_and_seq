import torch.nn as nn
import torch.utils.data as data

from models import BaseModel, WindowTagger, WindowModelWithPreTrainedEmbeddings, WindowModelWithSubWords, AcceptorLSTM, BasicBiLSTM, BiLSTMWithSubWords
from mappers import BaseMapper, TokenMapper, TokenMapperUnkCategory, TokenMapperWithSubWords, BaseMapperWithPadding, RegularLanguageMapper, TokenMapperUnkCategoryWithPadding, TokenMapperWithSubWordsWithPadding
from predictors import BasePredictor, WindowModelPredictor, WindowNERTaggerPredictor, AcceptorPredictor, GreedyLSTMPredictor, GreedyLSTMPredictorForNER
from configs import BaseConfig, ModelConfig, TrainingConfig, WindowTaggerConfig, InferenceConfig, RNNConfig
from datasets import WindowDataset, WindowWithSubWordsDataset, RegularLanguageDataset, BiLSTMDataset, BiLSTMWithSubWordsDataset
from trainers import ModelTrainer, AcceptorTrainer, BiLSTMTrainer


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

        if "lstm" in config_type:
            return RNNConfig()


class MappersFactory(object):
    def __call__(self, parameters_dict: BaseConfig, mapper_name: str) -> BaseMapper:

        # mapper_attributes
        min_frequency = parameters_dict["min_frequency"]
        split_char = parameters_dict["split_char"]

        if "window" in mapper_name:

            # check if flags exists
            if "smart_unknown" in parameters_dict:
                return TokenMapperUnkCategory(min_frequency, split_char)

            if "sub_word_units" in parameters_dict:
                return TokenMapperWithSubWords(min_frequency, split_char)

            return TokenMapper(min_frequency, split_char)

        if "lstm" in mapper_name:
            if "sub_words" in mapper_name:
                return TokenMapperWithSubWordsWithPadding(min_frequency, split_char)
            else:
                return TokenMapperUnkCategoryWithPadding(min_frequency, split_char)

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

        elif mapper_name == "TokenMapperUnkCategoryWithPadding":
            return TokenMapperUnkCategoryWithPadding()

        elif mapper_name == "TokenMapperWithSubWordsWithPadding":
            return TokenMapperWithSubWordsWithPadding()

        else:
            raise AttributeError("Wrong mapper name")


class ModelsFactory(object):

    def __call__(self, parameters_dict: BaseConfig, model_config: ModelConfig, mapper: BaseMapper, model_name: str) -> BaseModel:

        if "window" in model_name:
            # flags
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

        if "lstm" in model_name:
            model_config: RNNConfig

            if "sub_words" in model_name:
                mapper: TokenMapperWithSubWordsWithPadding
                return BiLSTMWithSubWords(model_config, mapper)
            else:
                mapper: BaseMapperWithPadding
                return BasicBiLSTM(model_config, mapper)

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

        elif model_name == "BasicBiLSTM":
            return BasicBiLSTM(model_config, mapper)

        elif model_name == "BiLSTMWithSubWords":
            return BiLSTMWithSubWords(model_config, mapper)

        else:
            raise AttributeError("Wrong model name")


class PredictorsFactory(object):
    def __call__(self, parameters_dict: BaseConfig, mapper: BaseMapper, predictor_type: str) -> BasePredictor:

        if predictor_type == "window_ner":
            return WindowNERTaggerPredictor(mapper)

        elif predictor_type == "window_pos":
            return WindowModelPredictor(mapper)

        elif predictor_type == "acceptor":
            return AcceptorPredictor(mapper)

        elif "lstm" in predictor_type:
            mapper: BaseMapperWithPadding

            if "_ner" in predictor_type:
                return GreedyLSTMPredictorForNER(mapper)

            if "_pos" in predictor_type:
                return GreedyLSTMPredictor(mapper)


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

        if "lstm" in dataset_type:
            sequence_length = parameters_dict["sequence_length"]
            if "sub_words" in dataset_type:
                mapper: TokenMapperWithSubWordsWithPadding
                return BiLSTMWithSubWordsDataset(file_path, mapper, sequence_length)

            else:
                mapper: BaseMapperWithPadding
                return BiLSTMDataset(file_path, mapper, sequence_length)

        if dataset_type == "acceptor":
            sequence_length = parameters_dict["sequence_length"]
            mapper: BaseMapperWithPadding
            return RegularLanguageDataset(file_path, mapper, sequence_length)

    def get_from_dataset_name(self, dataset_name, file_path, mapper, window_size=2, sequence_length=65):

        if dataset_name == "WindowWithSubWordsDataset":
            return WindowWithSubWordsDataset(file_path, mapper, window_size)

        elif dataset_name == "WindowDataset":

            return WindowDataset(file_path, mapper, window_size)

        elif dataset_name == "RegularLanguageDataset":
            return RegularLanguageDataset(file_path, mapper, sequence_length)

        elif dataset_name == "BiLSTMDataset":
            return BiLSTMDataset(file_path, mapper, sequence_length)

        elif dataset_name == "BiLSTMWithSubWordsDataset":
            return BiLSTMWithSubWordsDataset(file_path, mapper, sequence_length)

        else:
            raise AttributeError("Wrong dataset name")


class TrainerFactory(object):
    def __call__(self, model: BaseModel, train_config: TrainingConfig,
                 predictor: BasePredictor, loss_function: nn.Module,
                 model_type: str) -> ModelTrainer:

        if "window" in model_type:
            return ModelTrainer(model, train_config, predictor, loss_function)

        if "lstm" in model_type:
            return BiLSTMTrainer(model, train_config, predictor, loss_function)

        if model_type == "acceptor":
            return AcceptorTrainer(model, train_config, predictor, loss_function)


class LossFunctionFactory(object):
    def __call__(self, model_type: str, mapper: BaseMapper = None) -> nn.Module:

        if "window" in model_type:
            return nn.CrossEntropyLoss()

        if model_type == "acceptor":
            return nn.CrossEntropyLoss()

        if "lstm" in model_type:
            mapper: BaseMapperWithPadding
            padding_symbol = mapper.get_padding_symbol()
            label_padding_index = mapper.get_label_idx(padding_symbol)
            return nn.CrossEntropyLoss(ignore_index=label_padding_index)



