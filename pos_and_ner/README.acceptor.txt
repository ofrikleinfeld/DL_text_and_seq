The script "experiment.py" let you train an acceptor LSTM on different languages datasets and view the test set accuracy

## Usage

### Dependency Packages
The project has a standard "requirements.txt" file
You can use pip with the flag -r requirements.txt to download the required dependencies to your virtual environment

### Possible Commands
You can run this application as a command line utility:
```sh
$ python experiment.py --help
usage: experiment.py [-h] --name regex --train_path TRAIN_PATH --test_path
                     TEST_PATH --model_config_path MODEL_CONFIG_PATH
                     --training_config_path TRAINING_CONFIG_PATH

train and evaluate an acceptor model on train and test sets

optional arguments:
  -h, --help            show this help message and exit
  --name regex          unique name of the training procedure (used for
                        checkpoint saving
  --train_path TRAIN_PATH
                        a path to training file
  --test_path TEST_PATH
                        a path to test file
  --model_config_path MODEL_CONFIG_PATH
                        path to a json file containing model hyper parameters
  --training_config_path TRAINING_CONFIG_PATH
                        path to a json file containing hyper parameters for
                        training procedure
```
### Configuration Files:
For training you have to provide 2 configuration files containing all the required fields (specified in the example below)

The first configuration file is called "model config", and it has the following format:

```json
{
  "hidden_dim": 300
  ,"embedding_dim":50
}
```
- hidden_dim - size of hidden layer
- embedding_dim: size of embeddings

The second configuration file is called "model config: and it has the following format:
```json
{
    "batch_size": 64,
    "num_workers": 12,
    "device": "cuda",
    "num_epochs": 10,
    "print_step": 10,
    "learning_rate": 1e-2,
    "checkpoints_path": "experiments/acceptor_experiment",
    "min_frequency": 0,
    "split_char" : "\t",
    "sequence_length": 65
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
