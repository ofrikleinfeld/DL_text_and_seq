The script "bilstmTrain.py" and "bilstmPredict.py" let you train an LSTM model, save it, and use it later for inference of an unseen test data
You can choose between different embeddings/representations to use, before the LSTM layer, as we will detail shortly

## Usage

### Dependency Packages
The project has a standard "requirements.txt" file
You can use pip with the flag -r requirements.txt to download the required dependencies to your virtual environment

### Possible Commands
You can run both "bilstmTrain.py" and "bilstmPredict.py" as command line applications with the following arguments

```sh
$ python bilstmTrain.py --help

usage: bilstmTrain.py [-h] [--dev_path DEV_PATH]
                      {a,b,c,d} train_path model_path model_config_path
                      training_config_path {pos,ner}

train bilstm model

positional arguments:
  {a,b,c,d}             which type of word embeddings representation to use
  train_path            a path to training file
  model_path            a path to save the trained model
  model_config_path     path to a json file containing model hyper parameters
  training_config_path  path to a json file containing hyper parameters for
                        training procedure
  {pos,ner}             which model to run POS or NER

optional arguments:
  -h, --help            show this help message and exit
  --dev_path DEV_PATH   a path to dev file


$ python bilstmPredict.py --help

usage: bilstmPredict.py [-h]
                    {a,b,c,d} model_path test_path {pos,ner}
                    inference_config_path save_output_path

predict using a trained bilstm model

positional arguments:
  {a,b,c,d}             which type of word embeddings representation to use
  model_path            a path to a trained model file
  test_path             a path to a test file
  {pos,ner}             which model to run POS or NER
  inference_config_path
                        a path to inference config file
  save_output_path      a path to save the results of the inference on the
                        test set

optional arguments:
  -h, --help            show this help message and exit
```

### Details about the possible arguments
model type translation:
a) an embedding vector: repr(wi) = E[wi]
b) a character-level LSTM: repr(wi) = repr(c1, c2, ..., cmi) = LSTMC (E[c1], ..., E[cmi]).
c) embeddings + sub-word representation (prefix and suffix of 3 letters of each word)
d) a concatenation of (a) and (b) followed by a linear layer and a Tanh activation.

* model path argument:
in bilstmTrain script - this argument defines the unique name of the model run. please see also the "checkpoint_path" argument in training config to define root dir to save trained model
in bilstmTag script - this should be the exact path to the .pth file created during the training phase

### Configuration Files:
#### training:
For training you have to provide 2 configuration files containing all the required fields (specified in the example below)

The first configuration file is called "model config", and it has the following format:

```json
{
  "hidden_dim": 200,
  "embedding_dim":50,
  "char_hidden_dim": 50
}
```
- hidden_dim - size of hidden layer
- embedding_dim: size of embeddings
- char_hidden_dim: size of char embeddings to use in case of a character layer LSTM as embedding layer (option b)

The second configuration file is called "model config: and it has the following format:
```json
{
  "batch_size": 64,
  "num_workers": 12,
  "device": "cuda",
  "num_epochs": 10,
  "print_step": 500,
  "learning_rate": 1e-3,
  "checkpoints_path": "experiments/lstm_ner",
  "min_frequency": 5,
  "split_char" : "\t",
  "sequence_length": 50
}
```
- batch_size - the batch size used during training
- num_workers - num threads using for dataset loading
- device - should be one of the following ["cuda", "cuda:<gpu device id>", "cpu"]. indictaed which device to use for computation.
- num_epochs - number of training iterations
- print_step - number of batches between status prints and accuracy computation on dev set during training
- learning_rate - the learning rate used for training
- checkpoint_path - path to a directory where the trained model will be saved for future inference
- min_frequency - defines which words will be used as part training vocabulary. for example setting min_frequency to 5 means that only words appearing at least 5 times on the training set will be considered as seen in during training
- split_char: used for parsing the training data and separating between tokens and labels
- sequence length: define constant sequence length to each sentence during training to allow batching across examples. sentences longer than this number will be pruned and sentences shorter than this number will be padded

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