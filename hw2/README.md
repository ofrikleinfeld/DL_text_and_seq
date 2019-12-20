Ofri Kleinfeld

The code for this assignment lets you run a Pytorch application for training and running a tagging classifier, specificaly for NER/POS tasks. 

## Usage

### Dependency Packages
The project has a standard "requirements.txt" file
You can use pip with the flag -r requirements.txt to download the required dependencies to your virtual environment

### Possible Commands
You can run this application as a command line utility:
```sh
$ python main.py --help
usage: main.py [-h] {training,inference} ...

models training and prediction application


$ python main.py training --help

usage: main.py training [-h] --name window_ner --model_type
                        {window_ner,window_pos,window_pre_trained_ner,window_pre_trained_pos,window_sub_words_ner,window_sub_words_pos,window_pre_trained_sub_words_ner,window_pre_trained_sub_words_pos,acceptor,lstm_ner,lstm_pos,lstm_sub_words_ner,lstm_sub_words_pos}
                        --train_path TRAIN_PATH --dev_path DEV_PATH
                        --model_config_path MODEL_CONFIG_PATH
                        --training_config_path TRAINING_CONFIG_PATH

optional arguments:
  -h, --help            show this help message and exit
  --name window_ner     unique name of the training procedure (used for
                        checkpoint saving)
  --model_type {window_ner,window_pos,window_pre_trained_ner,window_pre_trained_pos,window_sub_words_ner,window_sub_words_pos,window_pre_trained_sub_words_ner,window_pre_trained_sub_words_pos,acceptor,lstm_ner,lstm_pos,lstm_sub_words_ner,lstm_sub_words_pos}
                        unique name of the training procedure (used for
                        checkpoint saving
  --train_path TRAIN_PATH
                        a path to training file
  --dev_path DEV_PATH   a path to a development set file
  --model_config_path MODEL_CONFIG_PATH
                        path to a json file containing model hyper parameters
  --training_config_path TRAINING_CONFIG_PATH
                        path to a json file containing hyper parameters for
                        training procedure

$ python main.py inference --help
usage: main.py inference [-h] --model_type
                         {window_ner,window_pos,window_pre_trained_ner,window_pre_trained_pos,window_sub_words_ner,window_sub_words_pos,window_pre_trained_sub_words_ner,window_pre_trained_sub_words_pos,acceptor,lstm_ner,lstm_pos,lstm_sub_words_ner,lstm_sub_words_pos}
                         --test_path TEST_PATH --trained_model_path
                         TRAINED_MODEL_PATH --save_output_path
                         SAVE_OUTPUT_PATH --inference_config_path
                         INFERENCE_CONFIG_PATH

optional arguments:
  -h, --help            show this help message and exit
  --model_type {window_ner,window_pos,window_pre_trained_ner,window_pre_trained_pos,window_sub_words_ner,window_sub_words_pos,window_pre_trained_sub_words_ner,window_pre_trained_sub_words_pos,acceptor,lstm_ner,lstm_pos,lstm_sub_words_ner,lstm_sub_words_pos}
                        unique name of the training procedure (used for
                        checkpoint saving
  --test_path TEST_PATH
                        a path to a test set file
  --trained_model_path TRAINED_MODEL_PATH
                        a path to a trained model checkpoint (.pth file
  --save_output_path SAVE_OUTPUT_PATH
                        a path to a save the prediction results
  --inference_config_path INFERENCE_CONFIG_PATH
                        path to a json file containing model hyper parameters
                        for inference procedure
```

the TRAINED_MODEL_PATH should be a path to a directory containing serialized trained model, that was created during a training process.

### Configuration Files:
#### training:
For training you have to provide 2 configuration files containing all the required fields (specified in the example below)

The first configuration file is called "model config", and it has the following format:

```json
{
  "hidden_dim": 200
  ,"embedding_dim":50
   ,"window_size": 2
}
```
- hidden_dim - size of hidden layer
- embedding_dim: size of embeddings
- window_size: number of words to consider before and after the given word (for window based taggers)

Please run training/inference --help and checkout the possible values of "model type" to understand the supported models
Currently there are two families of supported taggers - window-based model and sequence/RNN based models


The second configuration file is called "model config: and it has the following format:
```json
{
  "batch_size": 64,
  "num_workers": 12,
  "device": "cuda",
  "num_epochs": 10,
  "print_step": 500,
  "learning_rate": 1e-3,
  "checkpoints_path": "experiments/pos_simple",
  "min_frequency": 5,
  "split_char" : " "
}
```
- batch_size - the batch size used during training
- num_workers - num threads using for dataset loading
- device - should be one of the following ["cuda", "cuda:<gpu device id>", "cpu"]. indictaed which device to use for computation.
- num_epochs - number of training iterations
- print_step - number of batches between status prints during training
- learning_rate - the learning rate used for training
- checkpoint_path - path to a directory where the trained model will be saved for future inference
- min_frequency - defines which words will be used as part training vocabulary. for example setting min_frequency to 5 means that only words appearing at least 5 times on the training set will be considered as seen in during training
- split_char: used for parsing the training data and separating between tokens and labels
 
#### prediction/inference:
For prediction you have to provide 1 configuration file using the following format
```json
{
  "batch_size": 200
  ,"num_workers": 12
  ,"device": "cuda"
}
```
the attributes meanings are identical to those described in the training config file

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.