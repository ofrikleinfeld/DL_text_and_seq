import numpy as np
import torch
import torch.nn as nn
from configs import ModelConfig, WindowTaggerConfig
from mappers import BaseMapper


class BaseModel(nn.Module):

    def __init__(self, config: ModelConfig, mapper: BaseMapper):
        super().__init__()
        self.config = config
        self.mapper = mapper

    def serialize_model(self) -> dict:
        model_name = self.__class__.__name__
        config_class_name = self.config.__class__.__name__
        mapper_class_name = self.mapper.__class__.__name__

        config_params = self.config.to_dict()
        model_state = self.state_dict()
        mapper_state = self.mapper.serialize()

        model_state = {
            "model": {"name": model_name, "state": model_state},
            "config": {"name": config_class_name, "state": config_params},
            "mapper": {"name": mapper_class_name, "state": mapper_state},
        }

        return model_state


class WindowTagger(BaseModel):

    def __init__(self, config: WindowTaggerConfig, mapper: BaseMapper):
        super().__init__(config, mapper)
        embedding_dim = config.embedding_dim
        window_size = config.window_size
        tokens_dim = mapper.get_tokens_dim()
        labels_dim = mapper.get_labels_dim()
        hidden_dim = config.hidden_dim

        # layers
        input_dim = (2 * window_size + 1) * embedding_dim
        self.embedding = nn.Embedding(tokens_dim, embedding_dim)
        self.hidden = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.output = nn.Linear(in_features=hidden_dim, out_features=labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        window_embeddings = self.embedding(x)  # results in tensor of size (batch, window_size * 2 + 1, embedding_dim)
        embedding = torch.flatten(window_embeddings, start_dim=1)  # concatenate the embeddings of the window words and current word
        hidden = torch.tanh(self.hidden(embedding))
        y_hat = self.output(hidden)

        return y_hat


class ModelWithPreTrainedEmbeddings(BaseModel):

    def __init__(self, config: ModelConfig, mapper: BaseMapper):
        super().__init__(config, mapper)
        self.tokens_dim = mapper.get_tokens_dim()
        self.embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim)
        self.num_pre_trained_used = 0

    def load_pre_trained_embeddings(self, pre_trained_vocab_path: str, pre_trained_embedding_path: str):
        pre_trained_matrix: np.ndarray = np.loadtxt(pre_trained_embedding_path)
        embedding_matrix = self.embedding.weight.detach().numpy()
        pre_trained_vocab = self._load_pre_trained_vocab(pre_trained_vocab_path)
        mapper_vocab = self.mapper.token_to_idx

        for word, index in mapper_vocab.items():
            lower_word = word.lower()  # assign the pre-trained vector in a case-insensitive matter

            if lower_word in pre_trained_vocab:
                word_pre_trained_vector = pre_trained_matrix[pre_trained_vocab[lower_word]]
                # assign the pre-trained vector for the word
                embedding_matrix[index] = word_pre_trained_vector
                self.num_pre_trained_used += 1

        # load the new embedding matrix as the embedding layer parameters
        self.embedding.load_state_dict({'weight': torch.tensor(embedding_matrix)})

    def _load_pre_trained_vocab(self, vocab_path: str) -> dict:
        pre_trained_vocab = {}
        with open(vocab_path, "r", encoding="utf8") as f:
            for index, word in enumerate(f):
                word = word[:-1]  # remove end of line token
                pre_trained_vocab[word] = index

        return pre_trained_vocab


class WindowModelWithPreTrainedEmbeddings(ModelWithPreTrainedEmbeddings):
    def __init__(self, config: WindowTaggerConfig, mapper: BaseMapper, pre_trained_vocab_path: str,
                 pre_trained_embedding_path: str):
        super().__init__(config, mapper)
        self.load_pre_trained_embeddings(pre_trained_vocab_path, pre_trained_embedding_path)

        # define hyper parameters and layer
        window_size = config.window_size
        tokens_dim = mapper.get_tokens_dim()
        labels_dim = mapper.get_labels_dim()
        hidden_dim = config.hidden_dim

        # layers
        input_dim = (2 * window_size + 1) * self.embedding_dim
        self.embedding = nn.Embedding(tokens_dim, self.embedding_dim)
        self.hidden = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.output = nn.Linear(in_features=hidden_dim, out_features=labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        window_embeddings = self.embedding(x)  # results in tensor of size (batch, window_size * 2 + 1, embedding_dim)
        embedding = torch.flatten(window_embeddings, start_dim=1)  # concatenate the embeddings of the window words and current word
        hidden = torch.tanh(self.hidden(embedding))
        y_hat = self.output(hidden)

        return y_hat
