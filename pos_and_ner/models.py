import numpy as np
import torch
import torch.nn as nn
from pos_and_ner.configs import ModelConfig, WindowTaggerConfig, RNNConfig, RNNWithCharsEmbeddingsConfig, RNNWithCharsWithWordsEmbeddingsConfig
from pos_and_ner.mappers import BaseMapper, BaseMapperWithPadding, TokenMapperWithSubWords, TokenMapperWithSubWordsWithPadding, TokenMapperWithCharsWithWordsWithPadding, TokenMapperWithCharsWithPadding


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
        embedding_dim = config["embedding_dim"]
        window_size = config["window_size"]
        tokens_dim = mapper.get_tokens_dim()
        labels_dim = mapper.get_labels_dim()
        hidden_dim = config["hidden_dim"]

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
        self.embedding_dim = config["embedding_dim"]
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

        # define hyper parameters and layers
        window_size = config["window_size"]
        tokens_dim = mapper.get_tokens_dim()
        labels_dim = mapper.get_labels_dim()
        hidden_dim = config["hidden_dim"]

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


class WindowModelWithSubWords(ModelWithPreTrainedEmbeddings):
    def __init__(self, config: WindowTaggerConfig, mapper: TokenMapperWithSubWords, pre_trained: bool = False,
                 pre_trained_vocab_path: str = "", pre_trained_embedding_path: str = ""):
        super().__init__(config, mapper)
        if pre_trained:
            self.load_pre_trained_embeddings(pre_trained_vocab_path, pre_trained_embedding_path)

        # define hyper parameters and layers
        window_size = config["window_size"]
        hidden_dim = config["hidden_dim"]
        labels_dim = mapper.get_labels_dim()
        prefix_dim = mapper.get_prefix_dim()
        suffix_dim = mapper.get_suffix_dim()
        input_dim = (2 * window_size + 1) * self.embedding_dim

        self.prefix_embedding = nn.Embedding(prefix_dim, self.embedding_dim)
        self.suffix_embedding = nn.Embedding(suffix_dim, self.embedding_dim)
        self.hidden = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.output = nn.Linear(in_features=hidden_dim, out_features=labels_dim)

    def forward(self, x: list) -> torch.tensor:
        words_tokens = x[:, 0, :]
        prefix_tokens = x[:, 1, :]
        suffix_tokens = x[:, 2, :]
        word_embeddings = self.embedding(words_tokens)
        prefix_embeddings = self.prefix_embedding(prefix_tokens)
        suffix_embeddings = self.suffix_embedding(suffix_tokens)

        embeddings_sum = word_embeddings + prefix_embeddings + suffix_embeddings
        embedding = torch.flatten(embeddings_sum, start_dim=1)  # concatenate the embeddings of the window words and current word
        hidden = torch.tanh(self.hidden(embedding))
        y_hat = self.output(hidden)

        return y_hat


class AcceptorLSTM(BaseModel):
    def __init__(self, config: RNNConfig, mapper: BaseMapperWithPadding):
        super().__init__(config, mapper)
        self.tokens_dim = mapper.get_tokens_dim()
        self.labels_dim = mapper.get_labels_dim()
        self.padding_idx = mapper.get_padding_index()
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.padding_idx)
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            batch_first=True, bidirectional=False)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=self.labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        x = self.dropout(x)
        _, last_hidden = self.lstm(x)

        h_n, _ = last_hidden
        _, batch, features = h_n.size()
        h_n = h_n.view(batch, features)

        y_hat = self.linear(h_n)
        return y_hat


class BasicBiLSTM(BaseModel):
    def __init__(self, config: RNNConfig, mapper: BaseMapperWithPadding):
        super().__init__(config, mapper)
        self.tokens_dim = mapper.get_tokens_dim()
        self.labels_dim = mapper.get_labels_dim()
        self.padding_idx = mapper.get_padding_index()
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.padding_idx)
        self.LSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        rnn_features, _ = self.LSTM(x)
        # RNN outputs has dimensions batch, sequence_length, features (features is num_directions * hidden dim)

        y_hat = self.linear(rnn_features)

        # Cross Entropy loss expects dimensions of type batch, features, sequence
        y_hat = y_hat.permute(0, 2, 1)
        return y_hat


class BiLSTMWithSubWords(BaseModel):

    def __init__(self, config: RNNConfig, mapper: TokenMapperWithSubWordsWithPadding):
        super().__init__(config, mapper)
        self.tokens_dim = mapper.get_tokens_dim()
        self.labels_dim = mapper.get_labels_dim()
        self.prefix_dim = mapper.get_prefix_dim()
        self.suffix_dim = mapper.get_suffix_dim()
        self.padding_idx = mapper.get_padding_index()
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]

        self.word_embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.padding_idx)
        self.prefix_embedding = nn.Embedding(self.prefix_dim, self.embedding_dim, padding_idx=self.padding_idx)
        self.suffix_embedding = nn.Embedding(self.suffix_dim, self.embedding_dim, padding_idx=self.padding_idx)

        self.LSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:

        words_tokens = x[:, 0, :]
        prefix_tokens = x[:, 1, :]
        suffix_tokens = x[:, 2, :]

        word_embeddings = self.word_embedding(words_tokens)
        prefix_embeddings = self.prefix_embedding(prefix_tokens)
        suffix_embeddings = self.suffix_embedding(suffix_tokens)

        embeddings_sum = word_embeddings + prefix_embeddings + suffix_embeddings
        # embedding = torch.flatten(embeddings_sum, start_dim=1)

        rnn_features, _ = self.LSTM(embeddings_sum)
        # RNN outputs has dimensions batch, sequence_length, features (features is num_directions * hidden dim)

        y_hat = self.linear(rnn_features)

        # Cross Entropy loss expects dimensions of type batch, features, sequence
        y_hat = y_hat.permute(0, 2, 1)
        return y_hat


class BiLSTMWithChars(BaseModel):

    def __init__(self, config: RNNWithCharsEmbeddingsConfig, mapper: TokenMapperWithCharsWithPadding):
        super().__init__(config, mapper)
        self.tokens_dim = mapper.get_tokens_dim()
        self.labels_dim = mapper.get_labels_dim()
        self.padding_idx = mapper.get_padding_index()
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.char_hidden_dim = config["char_hidden_dim"]

        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.padding_idx)
        self.LSTM_c = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.char_hidden_dim,
                              num_layers=1, bidirectional=False, batch_first=True)
        self.LSTM = nn.LSTM(input_size=self.char_hidden_dim, hidden_size=self.hidden_dim,
                            num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:

        embeddings = self.embedding(x)
        batch, word_sequence, char_sequence, features = embeddings.size()

        # fold word sequence dimension into batch dimension
        embeddings = embeddings.view(batch * word_sequence, char_sequence, features)
        _, (chars_last_hidden, _) = self.LSTM_c(embeddings)

        # now expand back
        word_embeddings = chars_last_hidden.view(batch, word_sequence, self.char_hidden_dim)

        # now we are in the same situation as always - batch, sequence, features
        rnn_features, _ = self.LSTM(word_embeddings)

        # RNN outputs has dimensions batch, sequence_length, features (features is num_directions * hidden dim)
        y_hat = self.linear(rnn_features)

        # Cross Entropy loss expects dimensions of type batch, features, sequence
        y_hat = y_hat.permute(0, 2, 1)
        return y_hat


class BiLSTMWithCharsAndWords(BaseModel):

    def __init__(self, config: RNNWithCharsWithWordsEmbeddingsConfig, mapper: TokenMapperWithCharsWithWordsWithPadding):
        super().__init__(config, mapper)
        self.chars_dim = mapper.get_chars_dim()
        self.words_dim = mapper.get_tokens_dim()
        self.labels_dim = mapper.get_labels_dim()
        self.char_padding_idx = mapper.get_char_padding_index()
        self.padding_idx = mapper.get_padding_index()
        self.word_embedding_dim = config["embedding_dim"]
        self.char_embedding_dim = config["char_embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.char_hidden_dim = config["char_hidden_dim"]
        self.linear_embeds_out_dim = config["linear_embeds_out_dim"]

        self.chars_embedding = nn.Embedding(self.chars_dim, self.char_embedding_dim,  padding_idx=self.char_padding_idx)
        self.words_embedding = nn.Embedding(self.words_dim,  self.word_embedding_dim, padding_idx=self.padding_idx)

        self.LSTM_c = nn.LSTM(input_size=self.char_embedding_dim, hidden_size=self.char_hidden_dim,
                              num_layers=1, bidirectional=False, batch_first=True)
        self.liner_embeds = nn.Linear(in_features=(self.word_embedding_dim + self.char_hidden_dim), out_features=self.linear_embeds_out_dim)
        self.tanh = nn.Tanh()
        self.LSTM = nn.LSTM(input_size=self.linear_embeds_out_dim, hidden_size=self.hidden_dim,
                            num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)

        self.linear = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        chars_x = x[:, :, :-1]
        words_x = x[:, :, -1]

        char_embeddings = self.chars_embedding(chars_x)
        batch, word_sequence, char_sequence, features = char_embeddings.size()

        # fold word sequence dimension into batch dimension
        char_embeddings = char_embeddings.view(batch * word_sequence, char_sequence, features)
        _, (chars_last_hidden, _) = self.LSTM_c(char_embeddings)

        # now expand back
        word_chars_embeddings = chars_last_hidden.view(batch, word_sequence, self.char_hidden_dim)

        word_embeddings = self.words_embedding(words_x)

        concat_word_embeddings = torch.cat([word_chars_embeddings,word_embeddings], dim=2)

        concat_word_embeddings = self.tanh(self.liner_embeds(concat_word_embeddings))

        # now we are in the same situation as always - batch, sequence, features
        rnn_features, _ = self.LSTM(concat_word_embeddings)

        # RNN outputs has dimensions batch, sequence_length, features (features is num_directions * hidden dim)
        y_hat = self.linear(rnn_features)

        # Cross Entropy loss expects dimensions of type batch, features, sequence
        y_hat = y_hat.permute(0, 2, 1)
        return y_hat
