import json


class BaseConfig(object):
    """
    Configuration class to store the configuration objects for various use cases.
    """

    def __init__(self, config_dict: dict = None):
        if config_dict is not None:
            self.from_dict(config_dict)

    def from_dict(self, parameters: dict):
        """Constructs a `Config` from a Python dictionary of parameters."""
        for key, value in parameters.items():
            self.__dict__[key] = value

        return self

    def from_json_file(self, json_file: str):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
            parameters_dict = json.loads(text)

        return self.from_dict(parameters_dict)

    def to_dict(self) -> dict:
        """Serializes this instance to a Python dictionary."""
        output = self.__dict__
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str) -> None:
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class ModelConfig(BaseConfig):
    """
    Config class used to define model configuration
    """
    def __init__(self, config_dict=None, embedding_dim: int = 50):
        super().__init__(config_dict)
        if config_dict is None:
            self.embedding_dim = embedding_dim


class WindowTaggerConfig(ModelConfig):
    """
    Dedicated config class for the Window based tagger model
    """
    def __init__(self, config_dict=None,
                 embedding_dim: int = 50, hidden_dim: int = 500,
                 window_size: int = 2):
        super().__init__(config_dict, embedding_dim)
        if config_dict is None:
            self.embedding_dim = embedding_dim
            self.hidden_dim = hidden_dim
            self.window_size = window_size


class TrainingConfig(BaseConfig):
    """
    Configuration class to store training and data loading configurations.
    """
    def __init__(self, config_dict=None, batch_size: int = 16, num_workers: int = 12,
                 device: str = "cpu", num_epochs: int = 30, learning_rate: float = 1e-4,
                 checkpoints_path: str = "checkpoints", checkpoint_step: int = 10):
        super().__init__(config_dict)

        if config_dict is None:
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.device = device
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.checkpoints_path = checkpoints_path
            self.checkpoints_step = checkpoint_step
