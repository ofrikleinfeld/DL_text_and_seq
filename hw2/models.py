import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import ModelConfig, WindowTaggerConfig
from mappers import TokenMapper


class BaseModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def serialize_model(self) -> dict:
        config_dict = self.config.to_dict()
        state_dict = self.state_dict()

        model_state = {
            "state_dict": state_dict,
            "config_dict": config_dict
        }

        return model_state

    def deserialize_model(self, model_state: dict) -> None:
        self.load_state_dict(model_state)


class WindowTagger(BaseModel):

    def __init__(self, config: WindowTaggerConfig, mapper: TokenMapper):
        super().__init__(config)
        self.mapper = mapper
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
        hidden = F.tanh(self.hidden(embedding))
        y_hat = self.output(hidden)

        return y_hat

    def serialize_model(self) -> dict:
        model_state = super().serialize_model()
        mapper_dict = self.mapper.serialize()

        model_state["mapper_state"] = mapper_dict

        return model_state

    @classmethod
    def deserialize_model(cls, model_state: dict):
        state_dict: dict = model_state["state_dict"]
        mapper_state: dict = model_state["mapper_state"]
        config_dict: dict = model_state["config_dict"]

        mapper = TokenMapper.deserialize(mapper_state)
        config = WindowTaggerConfig(config_dict)

        model = cls(config, mapper)
        model.load_state_dict(state_dict)

        return model
