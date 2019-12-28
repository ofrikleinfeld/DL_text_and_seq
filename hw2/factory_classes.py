import torch.nn as nn
import torch.utils.data as data

from models import BaseModel, WindowTagger, WindowModelWithPreTrainedEmbeddings, WindowModelWithSubWords, AcceptorLSTM, BasicBiLSTM, BiLSTMWithSubWords, BiLSTMWithChars, BiLSTMWithCharsAndWords
from mappers import BaseMapper, TokenMapperUnkCategory, TokenMapperWithSubWords, BaseMapperWithPadding, RegularLanguageMapper, TokenMapperUnkCategoryWithPadding, TokenMapperWithSubWordsWithPadding, TokenMapperWithCharsWithWordsWithPadding, TokenMapperWithCharsWithPadding
from predictors import BasePredictor, WindowModelPredictor, WindowNERTaggerPredictor, AcceptorPredictor, GreedyLSTMPredictor, GreedyLSTMPredictorForNER
from configs import BaseConfig, ModelConfig, TrainingConfig, WindowTaggerConfig, InferenceConfig, RNNConfig, RNNWithCharsEmbeddingsConfig, RNNWithCharsWithWordsEmbeddingsConfig
from datasets import WindowDataset, WindowWithSubWordsDataset, RegularLanguageDataset, BiLSTMDataset, BiLSTMWithSubWordsDataset, BiLSTMWithCharsDataset, BiLSTMWithCharsAndWordDataset
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
            if "char_word_embeddings" in config_type:
                return RNNWithCharsWithWordsEmbeddingsConfig()
            elif "char_embeddings" in config_type:
                return RNNWithCharsEmbeddingsConfig()

            return RNNConfig()


class MappersFactory(object):
    def __call__(self, config: BaseConfig, mapper_name: str) -> BaseMapper:

        # mapper_attributes
        if "min_frequency" in config:
            min_frequency = config["min_frequency"]
        else:
            min_frequency = 0

        if "split_char" in config:
            split_char = config["split_char"]
        else:
            split_char = "\t"

        if "window" in mapper_name:

            if "sub_words" in mapper_name:
                return TokenMapperWithSubWords(min_frequency, split_char)

            else:
                return TokenMapperUnkCategory(min_frequency, split_char)

        if "lstm" in mapper_name:

            if "sub_words" in mapper_name:
                return TokenMapperWithSubWordsWithPadding(min_frequency, split_char)

            elif "char_word_embeddings" in mapper_name:
                if "char_min_frequency" in config:
                    char_min_frequency = config["char_min_frequency"]
                else:
                    char_min_frequency = 0
                return TokenMapperWithCharsWithWordsWithPadding(min_frequency, split_char, char_min_frequency)

            elif "char_embeddings" in mapper_name:
                return TokenMapperWithCharsWithPadding(min_frequency, split_char)

            else:
                return TokenMapperUnkCategoryWithPadding(min_frequency, split_char)

        if mapper_name == "acceptor":

            return RegularLanguageMapper(min_frequency, split_char)


class ModelsFactory(object):

    def __call__(self, parameters_dict: BaseConfig, model_config: ModelConfig, mapper: BaseMapper, model_name: str) -> BaseModel:

        if "window" in model_name:
            # flags
            model_config: WindowTaggerConfig
            with_sub_words = "sub_words" in model_name
            with_pre_training = "pre_trained" in model_name

            # both exists
            if with_sub_words and with_pre_training:
                mapper: TokenMapperWithSubWords
                return WindowModelWithSubWords(model_config, mapper, pre_trained=True,
                                               pre_trained_vocab_path="vocab.txt",
                                               pre_trained_embedding_path="wordVectors.txt")
            # none of them exists
            if not with_sub_words and not with_pre_training:
                model_config: WindowTaggerConfig
                return WindowTagger(model_config, mapper)

            # just sub words exists
            if with_sub_words and not with_pre_training:
                mapper: TokenMapperWithSubWords
                return WindowModelWithSubWords(model_config, mapper, pre_trained=False,
                                               pre_trained_vocab_path="vocab.txt",
                                               pre_trained_embedding_path="wordVectors.txt")

            # just pre training exists
            if not with_sub_words and with_pre_training:
                return WindowModelWithPreTrainedEmbeddings(model_config, mapper,
                                                           pre_trained_vocab_path="vocab.txt",
                                                           pre_trained_embedding_path="wordVectors.txt")

        if "lstm" in model_name:
            model_config: RNNConfig
            with_sub_words = "sub_words" in model_name

            if with_sub_words:
                mapper: TokenMapperWithSubWordsWithPadding
                return BiLSTMWithSubWords(model_config, mapper)
            elif "char_word_embeddings" in model_name:
                mapper:TokenMapperWithCharsWithWordsWithPadding
                return BiLSTMWithCharsAndWords(model_config, mapper)
            elif "char_embeddings" in model_name:
                mapper: TokenMapperWithCharsWithPadding
                return BiLSTMWithChars(model_config, mapper)

            else:
                mapper: BaseMapperWithPadding
                return BasicBiLSTM(model_config, mapper)

        if model_name == "acceptor":
            model_config: RNNConfig
            mapper: RegularLanguageMapper
            return AcceptorLSTM(model_config, mapper)


class PredictorsFactory(object):
    def __call__(self, parameters_dict: BaseConfig, mapper: BaseMapper, predictor_type: str) -> BasePredictor:

        if "window" in predictor_type:
            if "_ner" in predictor_type:
                return WindowNERTaggerPredictor(mapper)

            if "pos_" in predictor_type:
                return WindowModelPredictor(mapper)

        elif "lstm" in predictor_type:
            mapper: BaseMapperWithPadding

            if "_ner" in predictor_type:
                return GreedyLSTMPredictorForNER(mapper)

            if "_pos" in predictor_type:
                return GreedyLSTMPredictor(mapper)

        elif predictor_type == "acceptor":
            return AcceptorPredictor(mapper)


class DatasetsFactory(object):
    def __call__(self, config: BaseConfig, file_path: str, mapper: BaseMapper, dataset_type: str) -> data.Dataset:

        if "window" in dataset_type:

            if "window_size" in config:
                window_size = config["window_size"]
            else:
                window_size = 2  # default value

            if "sub_words" in dataset_type:
                mapper: TokenMapperWithSubWords
                return WindowWithSubWordsDataset(file_path, mapper, window_size)

            else:
                return WindowDataset(file_path, mapper, window_size)

        if "lstm" in dataset_type:

            if "sequence_length" in config:
                sequence_length = config["sequence_length"]
            else:
                sequence_length = 50  # default value

            if "sub_words" in dataset_type:
                mapper: TokenMapperWithSubWordsWithPadding
                return BiLSTMWithSubWordsDataset(file_path, mapper, sequence_length)

            if "char_word_embeddings" in dataset_type:
                if "char_sequence_length" in config:
                    char_sequence_length = config["char_sequence_length"]
                else:
                    char_sequence_length = 10

                mapper: TokenMapperWithCharsWithWordsWithPadding
                return BiLSTMWithCharsAndWordDataset(file_path, mapper, sequence_length, char_sequence_length)

            if "char_embeddings" in dataset_type:
                if "char_sequence_length" in config:
                    char_sequence_length = config["char_sequence_length"]
                else:
                    char_sequence_length = 10

                mapper: TokenMapperWithCharsWithPadding
                return BiLSTMWithCharsDataset(file_path, mapper, sequence_length, char_sequence_length)

            else:
                mapper: BaseMapperWithPadding
                return BiLSTMDataset(file_path, mapper, sequence_length)

        if dataset_type == "acceptor":

            if "sequence_length" in config:
                sequence_length = config["sequence_length"]
            else:
                sequence_length = 65  # default value

            mapper: BaseMapperWithPadding
            return RegularLanguageDataset(file_path, mapper, sequence_length)


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

        if "lstm" in model_type:
            mapper: BaseMapperWithPadding
            padding_symbol = mapper.get_padding_symbol()
            label_padding_index = mapper.get_label_idx(padding_symbol)
            return nn.CrossEntropyLoss(ignore_index=label_padding_index)

        if model_type == "acceptor":
            return nn.CrossEntropyLoss()

