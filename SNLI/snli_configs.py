from pos_and_ner.configs import ModelConfig


class SNLIDecomposeAttentionVanillaConfig(ModelConfig):
    """
    Config class used to define model configuration
    """
    def __init__(self, config_dict=None,  embedding_dim: int = 300, hidden_dim: int = 200):
        super().__init__(config_dict)
        if config_dict is None:
            self.config["embedding_dim"] = embedding_dim
            self.config["hidden_dim"] = hidden_dim
