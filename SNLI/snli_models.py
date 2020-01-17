import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_and_ner.models import ModelWithPreTrainedEmbeddings, BaseModel
from SNLI.snli_configs import SNLIDecomposeAttentionVanillaConfig
from SNLI.snli_mappers import SNLIMapperWithGloveIndices


class SNLIDecomposeAttentionVanillaModel(ModelWithPreTrainedEmbeddings):
    def __init__(self, config: SNLIDecomposeAttentionVanillaConfig, mapper: SNLIMapperWithGloveIndices, glove_path: str = None):
        super().__init__(config, mapper)

        # re-initialize to embedding layer to incorporate padding index
        self.tokens_dim = mapper.get_tokens_dim()
        print(f"Number of vocabulary tokens is: {self.tokens_dim}")
        embedding_dim = config["embedding_dim"]
        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.mapper.get_padding_index())

        # load glove pre-trained embeddings if needed
        if glove_path is not None:
            self._load_pre_trained_glove(glove_path)

        # mark the embedding layer as fixed, with no gradient updates
        self.embedding.weight.requires_grad = False

        # define layers sizes according to dataset and paper
        labels_dim = mapper.get_labels_dim()
        hidden_dim = config["hidden_dim"]

        # define layer or MLP composed from few layers
        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.mlp_f = self._define_mlp_layer(hidden_dim, hidden_dim)
        self.mlp_g = self._define_mlp_layer(2 * hidden_dim, hidden_dim)
        self.mlp_h = self._define_mlp_layer(2 * hidden_dim, hidden_dim)
        self.classification_layer = nn.Linear(hidden_dim, labels_dim, bias=True)

    def _load_pre_trained_glove(self, glove_path: str) -> None:
        self.mapper: SNLIMapperWithGloveIndices
        embedding_matrix = self.embedding.weight.detach().numpy()

        glove_words_sorted_indices = sorted(self.mapper.word_to_glove_idx.items(), key=lambda x: x[1])
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
                word_data_vocab_index = self.mapper.token_to_idx[word]
                embedding_matrix[word_data_vocab_index] = vector

                # update to next word appearing in glove
                current_glove_index += 1

        # load the new embedding matrix as the embedding layer parameters
        self.embedding.load_state_dict({'weight': torch.tensor(embedding_matrix)})

    def _define_mlp_layer(self, input_dim: int, output_dim: int) -> nn.Sequential:
        layers = [
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim, output_dim, bias=True),
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def _get_attention_weights(self, input1: torch.tensor, input2: torch.tensor) -> torch.tensor:
        raw_weights = torch.einsum("bij,bkj->bik", input1, input2)
        weights = F.softmax(raw_weights, dim=2)

        return weights

    def forward(self, sent1: torch.tensor, sent2: torch.tensor) -> torch.tensor:
        # embeddings
        sent1_embeddings = self.embedding(sent1)
        sent2_embeddings = self.embedding(sent2)

        # projection to hidden space
        a = self.embedding_projection(sent1_embeddings)
        b = self.embedding_projection(sent2_embeddings)

        # F MLP
        f1 = self.mlp_f(a)
        f2 = self.mlp_f(b)

        # attention phase (computing beta and alpha)
        beta_weights = self._get_attention_weights(f1, f2)
        beta = torch.einsum("bij,bjk->bik", beta_weights, b)
        alpha_weights = self._get_attention_weights(f2, f1)
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

        # classification
        output = self.classification_layer(hidden_output)

        return output
