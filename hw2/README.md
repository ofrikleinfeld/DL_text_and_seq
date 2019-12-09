HW2 Ofri Kleinfeld

The code for this assignment lets you run a Pytorch application for training and running a tagging classifier, specificaly for NER/POS tasks. 

## Usage

### Possible Commands
You can run this application as a command line utility:
```sh
$ python tagger1.py --help
usage: tagger1.py [-h] {training,inference} ...

NER/POS models training and prediction application


$ python tagger1.py training --help

usage: tagger1.py training [-h] --name ner_tagger --train_path TRAIN_PATH
                           --dev_path DEV_PATH
                           [--model_config_path MODEL_CONFIG_PATH]
                           [--training_config_path TRAINING_CONFIG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --name ner_tagger     unique name of the training procedure (used for
                        checkpoint saving
  --train_path TRAIN_PATH  a path to training file
  --dev_path DEV_PATH   a path to a development set file
  --model_config_path MODEL_CONFIG_PATH path to a json file containing model hyper parameters
  --training_config_path TRAINING_CONFIG_PATH path to a json file containing hyper parameters for training procedure

$ python tagger1.py inference --help
usage: tagger1.py inference [-h] --test_path TEST_PATH --trained_model_path
                            TRAINED_MODEL_PATH
                            [--inference_config_path INFERENCE_CONFIG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --test_path TEST_PATH
                        a path to a test set file
  --trained_model_path TRAINED_MODEL_PATH a path to a trained model checkpoint
  --inference_config_path INFERENCE_CONFIG_PATH
                        path to a json file containing model hyper parameters
                        for inference procedure
```

the TRAINED_MODEL_PATH should be a path to a directory containing serialized trained model, that was created during a training process.

### Configuration Files:
#### training:
For training you have to provide 2configuration files containing all the required fields (specified in the example below)

The first configuration file is called "model config", and it has the following format:

```json
{
  "hidden_dim": 200
  ,"embedding_dim":50
    ,"window_size": 2
}
```
- hideen_dim - size of hidden layer
- embedding_dim: size of embeddings
- window_size: number of words to consider before and after the given word

Currently the project supports only a window tagger with 1 hidden layer. 


The second configuration file is called "model config: and it has the following format:
```json
{
  "model_type": "pos",
  "batch_size": 64,
  "num_workers": 12,
  "device": "cuda",
  "num_epochs": 10,
  "print_step": 500,
    "learning_rate": 1e-3,
  "checkpoints_path": "experiments/pos_simple",
  "min_frequency": 5,
  "split_char" : " ",
  "pre_trained_embeddings": "false",
  "sub_word_units": "false",
  "smart_unknown": "true"
}
```
- model_type - should be only ["pos, "ner"]. it is important for proper accuracy computation for NER model
- batch_size - the batch size used during training
- num_workers - num threads using for dataset loading
- device - should be one of the following ["cuda", "cuda:<gpu device id>", "cpu"]. indictaed which device to use for computation.
- num_epochs - number of training iterations
- print_step - number of batches between status prints during training
- learning_rate - the learning rate used for training
- checkpoint_path - path to a directory where the trained model will be saved for future inference
- min_frequency - defines which words will be used as part training vocabolary. for example setting min_frequency to 5 means that only words appearing at least 5 times on the training set will be considered as seen in during training
- split_char: used for parsing the training data and separating between tokens and labels
- pre_trained embedding - should be one of ["true","false"]. if you want to use pre trained embeddings you need to have 2 files on your working dir, called vocab.txt (file of vocabolary tokens, one in a line) and wordVectors.txt (file of actual embeddings, each line is a vector)
- sub_word units - whether to use additinal embeddigs of sub word units (3 first letters and 3 last letters of each token)
- smart_unknown - whether to group unknown words during training to groups such as numbers/dates/all captial etc. if these option if off, on token "UNK" will be used for every word not part of the vocabolary during training.  
 
#### prediction/inference:
For prediction you have to provide 1 configuration file using the following format
```json
{
  "model_type": "ner",
  "batch_size": 32
  ,"num_workers": 12
  ,"device": "cuda"
}
```
the attributes meaning are identical to those described in the training config file

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.