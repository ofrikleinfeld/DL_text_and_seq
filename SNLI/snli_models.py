from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pos_and_ner.models import ModelWithPreTrainedEmbeddings
from SNLI.snli_configs import SNLIDecomposeAttentionVanillaConfig
from SNLI.snli_mappers import SNLIMapperWithGloveIndices


class SNLIDecomposeAttentionVanillaModel(ModelWithPreTrainedEmbeddings):
    def __init__(self, config: SNLIDecomposeAttentionVanillaConfig, mapper: SNLIMapperWithGloveIndices,
                 pre_trained_vocab_path: str = None, pre_trained_embedding_path: str = None):
        super().__init__(config, mapper)

        # re-initialize to embedding layer to incorporate padding index
        self.tokens_dim = mapper.get_tokens_dim()
        embedding_dim = config["embedding_dim"]
        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.mapper.get_padding_index())

        # load glove pre-trained embeddings if needed
        if pre_trained_vocab_path is not None:
            self.load_pre_trained_embeddings(pre_trained_vocab_path, pre_trained_embedding_path)

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        sent1 = x[:, :, 0]
        sent2 = x[:, :, 1]

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

