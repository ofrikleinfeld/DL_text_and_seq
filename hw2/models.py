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
