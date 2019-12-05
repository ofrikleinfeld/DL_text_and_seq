import torch
from torch.utils import data
from mappers import BaseMapper


class BasePredictor(object):

    def __init__(self, mapper: BaseMapper):
        self.mapper = mapper

    def infer_model_outputs(self, model_outputs: torch.tensor):
        return NotImplementedError("A class deriving from BasePredictor must implement infer_model_outputs method")

    def infer_sample(self, model: torch.nn.Module, tokens_indices: torch.tensor):
        model_outputs = model(tokens_indices)
        return self.infer_model_outputs(model_outputs)

    def infer_raw_sample(self, model: torch.nn.Module, raw_sample: list):
        sample_tokens = []
        for sample in raw_sample:
            token_indices = [self.mapper.get_token_idx(token) for token in sample]
            sample_tokens.append(token_indices)

        sample_tokens = torch.tensor(sample_tokens)
        return self.infer_sample(model, sample_tokens)


class WindowModelPredictor(BasePredictor):

    def __init__(self, mapper: BaseMapper):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor):
        predictions = []
        _, labels_tokens = torch.max(model_outputs, dim=1)

        for i in range(len(model_outputs)):  # every sample in case of batch (even batch of size 1)
            current_prediction = labels_tokens[i].cpu().numpy()
            predicted_label = self.mapper.get_label_idx(current_prediction)
            predictions.append(predicted_label)

        return predictions
