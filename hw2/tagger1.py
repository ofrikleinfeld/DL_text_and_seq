from train_script import train
from inference_script import inference

if __name__ == '__main__':
    name = "ner_model"
    train_path = "ner/train"
    dev_path = "ner/dev"
    test_path = "ner/test"
    model_config_path = "window_tagger_config.json"
    training_config_path = "training_config.json"
    inference_config_path = "inference_config.json"

    mapper_name = "TokenMapperUnkCategory"
    model_config_name = "WindowTaggerConfig"
    model_name = "WindowTagger"
    train_config_name = "TrainingConfig"
    inference_config_name = "InferenceConfig"
    predictor_name = "WindowNERTaggerPredictor"

    # train(name, train_path, dev_path, model_config_name, model_config_path, train_config_name,
    #       training_config_path, model_name, mapper_name, predictor_name)

    trained_model_path = "C:\\Users\\t-ofklei\\Documents\\University\\DL_text_and_seq\\hw2\\checkpoints\\ner_model\\06-12-19_best_model.pth"
    predictions = inference(test_path, inference_config_name, inference_config_path, trained_model_path, predictor_name)
    print(len(predictions))
