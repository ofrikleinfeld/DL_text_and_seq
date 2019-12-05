import torch
from torch.utils import data
from mappers import TokenMapper


class BasePredictor(object):

    def __init__(self, mapper: TokenMapper):
        self.mapper = mapper

    def infer_sample(self, model: torch.nn.Module, tokenized_text: torch.tensor):
        return NotImplementedError("A class deriving from BasePredictor must implement infer_sample method")

    def infer_raw_sample(self, model: torch.nn.Module, raw_sample: list):
        return NotImplementedError("A class deriving from BasePredictor must implement infer_raw_sample method")

class WindowModelPredictor(BasePredictor):

    def __init__(self, mapper: TokenMapper):
        super().__init__(mapper)

    def infer_sample(self, model: torch.nn.Module, tokenized_text: torch.tensor):
        idx_to_label = self.mapper.idx_to_label
        predictions = []

