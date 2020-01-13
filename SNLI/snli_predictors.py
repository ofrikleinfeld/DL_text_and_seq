import torch

from pos_and_ner.predictors import BasePredictor
from pos_and_ner.mappers import BaseMapper


class SNLIPredictor(BasePredictor):

    def __init__(self, mapper: BaseMapper):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor) -> torch.tensor:
        _, labels_tokens = torch.max(model_outputs, dim=1)

        return labels_tokens
