
Ofri Kleinfeld, Omer Zalmanson

The code implenents the paper [A Decomposable Attention Model for Natural Language Inferences]([https://arxiv.org/pdf/1606.01933v1.pdf](https://arxiv.org/pdf/1606.01933v1.pdf)) appeared on EMNLP 2016

## Dependencies

### Dependency Packages
The project has a standard "requirements.txt" file
You can use pip with the flag -r requirements.txt to download the required dependencies to your virtual environment

### Glove pre-trained embeddings
The implementation uses Glove pre-trained embeddings.
Please download file [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip) and extract it
You can also browse to [https://nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) and download from there.

### Configuration files
In order to replicate the paper configurations, please use the following configuration files:

The model configuration file is shared for both configuration:

```json
{
  "hidden_dim": 200,
  "embedding_dim":300
}
```
The embedding dimension is determined by the pre-trained embeddings, you can change it if you would like to use other pre-trained embeddings
The hidden layer dimension it relevant for the MLPs in the architecture, matching the description in the paper

For the "vanilla" version please use the following training configuration file
```json
{  
   "batch_size": 32,  
   "num_workers": 12,  
   "device": "cuda",  
   "num_epochs": 200,  
   "print_step": 2500,  
   "learning_rate": 0.005,  
   "checkpoints_path": "SNLI/experiments",  
   "split_char": "\t",  
   "min_frequency": 0,  
   "sequence_length": 25,  
   "glove_path": "SNLI/glove.840B.300d.txt"  
}
```
For the intra-sentence attention version we will have to adapt the learning rate
```json
{  
   "batch_size": 32,  
   "num_workers": 12,  
   "device": "cuda",  
   "num_epochs": 200,  
   "print_step": 2500,  
   "learning_rate": 0.0015,  
   "checkpoints_path": "SNLI/experiments",  
   "split_char": "\t",  
   "min_frequency": 0,  
   "sequence_length": 25,  
   "glove_path": "SNLI/glove.840B.300d.txt"  
}
```
Here is a short explanation about the fields used in configuration:
- batch_size - the batch size used during training
- num_workers - num threads using for dataset loading
- device - should be one of the following ["cuda", "cuda:\<gpu device id\>", "cpu"]. indictaed which device to use for computation.
- num_epochs - number of training iterations
- print_step - number of batches between status prints during training
- learning_rate - the learning rate used for training (matching the one used in the paper)
- checkpoint_path - path to a directory where the trained model will be saved for future inference
- min_frequency - defines which words will be used as part training vocabulary. for example setting min_frequency to 0 means we will use every word appearing in the training set, if it also appears in Glove pre-trained vocabullary.
- sequence_length - what is the maximum sequence length used for embedding. In order to allow batching, all the samples will have the same length, shorter sentence will be padded and longer will be pruned.
- glove_path:  path to glove file extracted from zip archive 

### Running Experiments
You can run this application as a command line utility, using the main.py file and supplying the following arguments:

Vanilla model

```sh
$ python main.py training
--name <experiment_name>
--model_type SNLI_attention_vanilla
--train_path <path_to_snli_1.0_train.txt file> 
--dev_path <path_to_snli_1.0_test.txt file>
--model_config_path <path_to_model_config.json_file>
--training_config_path <path_to_training_config.json_file>
```
Intra-sentence model
```sh
$ python main.py training
--name <experiment_name>
--model_type SNLI_attention_intra_sent
--train_path <path_to_snli_1.0_train.txt file> 
--dev_path <path_to_snli_1.0_test.txt file>
--model_config_path <path_to_model_config.json_file>
--training_config_path <path_to_training_config.json_file>
```