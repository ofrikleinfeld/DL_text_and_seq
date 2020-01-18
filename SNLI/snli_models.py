from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_and_ner.models import ModelWithPreTrainedEmbeddings
from SNLI.snli_configs import SNLIDecomposeAttentionVanillaConfig
from SNLI.snli_mappers import SNLIMapperWithGloveIndices


class SNLIDecomposeAttentionMLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dropout_1 = nn.Dropout(p=0.2)
        self.linear_1 = nn.Linear(input_dim, output_dim, bias=True)
        self.relu_1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        self.linear_2 = nn.Linear(output_dim, output_dim, bias=True)
        self.relu_2 = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.dropout_1(x)
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        output = self.relu_2(x)

        return output


class SNLIDecomposeAttentionEncoderLayer(nn.Module):

    def __init__(self, tokens_dim: int, embedding_dim: int, hidden_dim: int, padding_index: int):
        super().__init__()
        self.embedding = nn.Embedding(tokens_dim, embedding_dim, padding_idx=padding_index)
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim, bias=True)

    def forward(self, sent1: torch.tensor, sent2: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # embeddings
        sent1_embeddings = self.embedding(sent1)
        sent2_embeddings = self.embedding(sent2)

        # projection to hidden space
        a = self.embedding_projection(sent1_embeddings)
        b = self.embedding_projection(sent2_embeddings)

        return a, b

    def get_embedding_layer(self) -> nn.Embedding:
        return self.embedding

    def set_embedding_layer(self, pre_trained_embedding_layer: nn.Embedding) -> None:
        self.embedding = pre_trained_embedding_layer


class SNLIDecomposeAttentionIntraSentenceEncoderLayer(SNLIDecomposeAttentionEncoderLayer):

    def __init__(self, tokens_dim: int, embedding_dim: int, hidden_dim: int, padding_index: int, sequence_length: int):
        super().__init__(tokens_dim, embedding_dim, hidden_dim, padding_index)
        self.f_intra = SNLIDecomposeAttentionMLP(hidden_dim, hidden_dim)
        self.dist_bias = torch.nn.Parameter(torch.randn(sequence_length), requires_grad=True)

    def _self_attention(self, x: torch.tensor) -> torch.tensor:
        batch_size, sequence_len, _ = x.size()
        distance_bias_matrix = self.dist_bias.expand(batch_size, sequence_len, sequence_len)

        f_x = self.f_intra(x)

        raw_weights = torch.einsum("bij,bkj->bik", f_x, f_x)
        distance_aware_raw_weights = raw_weights + distance_bias_matrix
        weights = F.softmax(distance_aware_raw_weights, dim=2)
        x_tag = torch.einsum("bij,bjk->bik", weights, x)

        return x_tag

    def forward(self, sent1: torch.tensor, sent2: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        a, b = super().forward(sent1, sent2)

        # self attention phase
        a_tag = self._self_attention(a)
        b_tag = self._self_attention(b)

        # concatenate on features axis
        a_a_tag = torch.cat([a, a_tag], dim=2)
        b_b_tag = torch.cat([b, b_tag], dim=2)

        return a_a_tag, b_b_tag


class SNLIDecomposeAttentionAttendCompareAggregateLayer(nn.Module):

    @staticmethod
    def get_attention_weights(input1: torch.tensor, input2: torch.tensor) -> torch.tensor:
        raw_weights = torch.einsum("bij,bkj->bik", input1, input2)
        weights = F.softmax(raw_weights, dim=2)

        return weights

    def __init__(self,
                 f_input_dim: int, f_output_dim: int,
                 g_input_dim: int, g_output_dim: int,
                 h_input_dim: int, h_output_dim: int):
        super().__init__()
        self.mlp_f = SNLIDecomposeAttentionMLP(f_input_dim, f_output_dim)
        self.mlp_g = SNLIDecomposeAttentionMLP(g_input_dim, g_output_dim)
        self.mlp_h = SNLIDecomposeAttentionMLP(h_input_dim, h_output_dim)

    def forward(self, a: torch.tensor, b: torch.tensor) -> torch.tensor:
        # F MLP
        f1 = self.mlp_f(a)
        f2 = self.mlp_f(b)

        # attention phase (computing beta and alpha)
        beta_weights = self.get_attention_weights(f1, f2)
        beta = torch.einsum("bij,bjk->bik", beta_weights, b)
        alpha_weights = self.get_attention_weights(f2, f1)
        alpha = torch.einsum("bij,bjk->bik", alpha_weights, a)

        # concatenation and G MLP
        # concatenate on features axis
        a_and_beta = torch.cat([a, beta], dim=2)
        b_and_alpha = torch.cat([b, alpha], dim=2)

        v1 = self.mlp_g(a_and_beta)
        v2 = self.mlp_g(b_and_alpha)

        # aggregation and H MLP
        # sum over sequence axis
        v1 = torch.sum(v1, dim=1)
        v2 = torch.sum(v2, dim=1)

        # again concatenate on features axis
        v1_and_v2 = torch.cat([v1, v2], dim=1)
        hidden_output = self.mlp_h(v1_and_v2)

        return hidden_output


class SNLIDecomposeAttentionVanillaModel(ModelWithPreTrainedEmbeddings):

    @staticmethod
    def load_pre_trained_glove(embedding_layer: nn.Embedding, word_to_idx: Dict[str, int], word_to_glove_idx: Dict[str, int], glove_path: str) -> nn.Embedding:
        embedding_matrix = embedding_layer.weight.detach().numpy()

        glove_words_sorted_indices = sorted(word_to_glove_idx.items(), key=lambda x: x[1])
        total_glove_words = len(glove_words_sorted_indices)
        current_glove_index = 0

        with open(glove_path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                # we read already all words in dictionary that are also in glove vocab
                if current_glove_index == total_glove_words:
                    break

                # skip until we find a word that also appears in vocab
                glove_word, glove_index = glove_words_sorted_indices[current_glove_index]
                if index != glove_index:
                    continue

                # retrieve pre-trained vector
                line = line[:-1]  # remove end of line
                line_tokens = line.split()
                word = line_tokens[0]
                vector = np.array(line_tokens[1:], dtype=np.float)

                # find out the corresponding index in our initiated embedding matrix  and update it
                word_data_vocab_index = word_to_idx[word]
                embedding_matrix[word_data_vocab_index] = vector

                # update to next word appearing in glove
                current_glove_index += 1

        # load the new embedding matrix as the embedding layer parameters
        embedding_layer.load_state_dict({'weight': torch.tensor(embedding_matrix)})
        return embedding_layer

    def __init__(self, config: SNLIDecomposeAttentionVanillaConfig, mapper: SNLIMapperWithGloveIndices, glove_path: str = None):
        super().__init__(config, mapper)

        # define dimensions of layers
        tokens_dim = mapper.get_tokens_dim()
        labels_dim = mapper.get_labels_dim()
        padding_index = mapper.get_padding_index()
        embedding_dim = config["embedding_dim"]
        hidden_dim = config["hidden_dim"]

        print(f"Number of vocabulary tokens is: {self.tokens_dim}")

        # define model sub components and layers
        self.encoder: SNLIDecomposeAttentionEncoderLayer = SNLIDecomposeAttentionEncoderLayer(tokens_dim, embedding_dim, hidden_dim, padding_index)
        self.attention_aggregation = SNLIDecomposeAttentionAttendCompareAggregateLayer(f_input_dim=hidden_dim, f_output_dim=hidden_dim,
                                                                                       g_input_dim=2 * hidden_dim, g_output_dim=hidden_dim,
                                                                                       h_input_dim=2 * hidden_dim, h_output_dim=hidden_dim)
        self.classification_layer = nn.Linear(hidden_dim, labels_dim, bias=True)

        # load pre trained glove embeddings, and mark embedding layer an not learned
        self.mapper: SNLIMapperWithGloveIndices
        embedding_layer = self.encoder.get_embedding_layer()
        word_to_index = self.mapper.token_to_idx
        word_to_glove_index = self.mapper.word_to_glove_idx
        if glove_path is not None:
            pre_trained_embedding_layer = self.load_pre_trained_glove(embedding_layer, word_to_index, word_to_glove_index, glove_path)
            pre_trained_embedding_layer.weight.requires_grad = False
            self.encoder.set_embedding_layer(pre_trained_embedding_layer)

    def forward(self, sent1: torch.tensor, sent2: torch.tensor) -> torch.tensor:
        a, b = self.encoder(sent1, sent2)
        hidden_output = self.attention_aggregation(a, b)
        output = self.classification_layer(hidden_output)

        return output


class SNLIDecomposeAttentionIntraSentenceModel(ModelWithPreTrainedEmbeddings):
    def __init__(self, config: SNLIDecomposeAttentionVanillaConfig, mapper: SNLIMapperWithGloveIndices, glove_path: str = None, sequence_length: int = 25):
        super().__init__(config, mapper)

        # define dimensions of layers
        tokens_dim = mapper.get_tokens_dim()
        labels_dim = mapper.get_labels_dim()
        padding_index = mapper.get_padding_index()
        embedding_dim = config["embedding_dim"]
        hidden_dim = config["hidden_dim"]

        print(f"Number of vocabulary tokens is: {self.tokens_dim}")

        # define model sub components and layers
        self.encoder: SNLIDecomposeAttentionEncoderLayer = SNLIDecomposeAttentionIntraSentenceEncoderLayer(tokens_dim, embedding_dim, hidden_dim, padding_index, sequence_length)
        self.attention_aggregation = SNLIDecomposeAttentionAttendCompareAggregateLayer(f_input_dim=2 * hidden_dim, f_output_dim=hidden_dim,
                                                                                       g_input_dim=4 * hidden_dim, g_output_dim=hidden_dim,
                                                                                       h_input_dim=2 * hidden_dim, h_output_dim=hidden_dim)
        self.classification_layer = nn.Linear(hidden_dim, labels_dim, bias=True)

        # load pre trained glove embeddings, and mark embedding layer an not learned
        self.mapper: SNLIMapperWithGloveIndices
        embedding_layer = self.encoder.get_embedding_layer()
        word_to_index = self.mapper.token_to_idx
        word_to_glove_index = self.mapper.word_to_glove_idx
        if glove_path is not None:
            pre_trained_embedding_layer = SNLIDecomposeAttentionVanillaModel.load_pre_trained_glove(embedding_layer, word_to_index, word_to_glove_index, glove_path)
            pre_trained_embedding_layer.weight.requires_grad = False
            self.encoder.set_embedding_layer(pre_trained_embedding_layer)

    def forward(self, sent1: torch.tensor, sent2: torch.tensor) -> torch.tensor:
        a, b = self.encoder(sent1, sent2)
        hidden_output = self.attention_aggregation(a, b)
        output = self.classification_layer(hidden_output)

        return output
