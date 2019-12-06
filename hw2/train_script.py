import torch.nn as nn
from configs import TrainingConfig, WindowTaggerConfig
from mappers import TokenMapperUnkCategory
from datasets import WindowDataset
from models import WindowTagger
from trainers import ModelTrainer
from predictors import WindowModelPredictor

if __name__ == '__main__':

    # paths to datasets
    train_path = "ner/train"
    dev_path = "ner/dev"

    # paths to json config files
    model_config_path = "window_tagger_config.json"
    training_config_path = "training_config.json"

    # create config object
    model_config = WindowTaggerConfig().from_json_file(model_config_path)
    training_config = TrainingConfig().from_json_file(training_config_path)

    # read training file and create a mapping from token to indices
    mapper = TokenMapperUnkCategory(min_frequency=5)
    mapper.create_mapping(train_path)

    # create dataset for training and dev
    train_data = WindowDataset(train_path, mapper)
    dev_data = WindowDataset(dev_path, mapper)

    # create a model
    window_tagger = WindowTagger(model_config, mapper)

    # create a trainer object, predictor, and a loss function
    # then start training
    ce_loss = nn.CrossEntropyLoss()
    window_predictor = WindowModelPredictor(mapper)
    trainer = ModelTrainer(window_tagger, training_config, window_predictor, ce_loss)
    trainer.train("NER", train_dataset=train_data, dev_dataset=dev_data)
