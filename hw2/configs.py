import json


class BaseConfig(object):
    """
    Configuration class to store the configuration objects for various use cases.
    """

    def __init__(self, config_dict: dict = None):
        self.config = {}
        if config_dict is not None:
            self.from_dict(config_dict)

    def __getitem__(self, item):
        return self.config[item]

    def add_key_value(self, key, value) -> None:
        self.config[key] = value

    def __contains__(self, item):
        return item in self.config

    def from_dict(self, parameters: dict):
        """Constructs a `Config` from a Python dictionary of parameters."""
        for key, value in parameters.items():

            if type(value) == str:
                value_lower_case = value.lower()

                if value_lower_case == "false":
                    value = False
                elif value_lower_case == "true":
                    value = True

            self.config[key] = value

        return self

    def from_json_file(self, json_file: str):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
            parameters_dict = json.loads(text)

        return self.from_dict(parameters_dict)

    def to_dict(self) -> dict:
        """Serializes this instance to a Python dictionary."""
        output = self.config
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
            self.config["embedding_dim"] = embedding_dim


class WindowTaggerConfig(ModelConfig):
    """
    Dedicated config class for the Window based tagger model
    """
    def __init__(self, config_dict=None, embedding_dim: int = 50, hidden_dim: int = 500, window_size: int = 2):
        super().__init__(config_dict, embedding_dim)
        if config_dict is None:
            self.config["hidden_dim"] = hidden_dim
            self.config["window_size"] = window_size


class RNNConfig(ModelConfig):

    def __init__(self, config_dict=None, embedding_dim: int = 50, hidden_dim: int = 150):
        super().__init__(config_dict, embedding_dim)
        if config_dict is None:
            self.config["hidden_dim"] = hidden_dim


class TrainingConfig(BaseConfig):
    """
    Configuration class to store training and data loading configurations.
    """
    def __init__(self, config_dict=None, model_type: str = "ner",
                 batch_size: int = 16, num_workers: int = 12,
                 device: str = "cpu", num_epochs: int = 30, learning_rate: float = 1e-4,
                 checkpoints_path: str = "checkpoints", checkpoint_step: int = 10,
                 print_step: int = 50):
        super().__init__(config_dict)

        if config_dict is None:
            self.config["model_type"] = model_type
            self.config["batch_size"] = batch_size
            self.config["num_workers"] = num_workers
            self.config["device"] = device
            self.config["num_epochs"] = num_epochs
            self.config["learning_rate"] = learning_rate
            self.config["checkpoints_path"] = checkpoints_path
            self.config["checkpoint_step"] = checkpoint_step
            self.config["print_step"] = print_step


class InferenceConfig(BaseConfig):
    def __init__(self, config_dict=None, batch_size: int = 16, num_workers: int = 12, device: str = "cpu"):
        super().__init__(config_dict)

        if config_dict is None:
            self.config["batch_size"] = batch_size
            self.config["num_workers"] = num_workers
            self.config["device"] = device
