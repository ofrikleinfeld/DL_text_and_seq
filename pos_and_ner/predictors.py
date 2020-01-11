from typing import Tuple

import torch

from pos_and_ner.mappers import BaseMapper, BaseMapperWithPadding


class BasePredictor(object):

    def __init__(self, mapper: BaseMapper):
        self.mapper = mapper

    def infer_model_outputs(self, model_outputs: torch.tensor) -> torch.tensor:
        raise NotImplementedError("A class deriving from BasePredictor must implement infer_model_outputs method")

    def infer_sample(self, model: torch.nn.Module, tokens_indices: torch.tensor):
        model_outputs = model(tokens_indices)
        raise self.infer_model_outputs(model_outputs)

    def infer_raw_sample(self, model: torch.nn.Module, raw_sample: list):
        sample_tokens = []
        for sample in raw_sample:
            token_indices = [self.mapper.get_token_idx(token) for token in sample]
            sample_tokens.append(token_indices)

        sample_tokens = torch.tensor(sample_tokens)
        return self.infer_sample(model, sample_tokens)

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> Tuple[int, int]:
        num_correct: int
        num_predictions: int

        num_predictions = len(labels)
        predictions: torch.tensor = self.infer_model_outputs(model_outputs)
        correct_predictions: torch.tensor = (predictions == labels).type(torch.int64)
        num_correct = torch.sum(correct_predictions).item()

        return num_correct, num_predictions


class WindowModelPredictor(BasePredictor):

    def __init__(self, mapper: BaseMapper):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor) -> torch.tensor:
        _, labels_tokens = torch.max(model_outputs, dim=1)

        return labels_tokens


class WindowNERTaggerPredictor(WindowModelPredictor):

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> (int, int):
        # special case of unbalanced labels
        # don't compute accuracy over trivial cases of the 'O' tag
        num_correct: int
        num_predictions: int

        predictions: torch.tensor = self.infer_model_outputs(model_outputs)
        O_tag_label = self.mapper.get_label_idx('O')
        labels_mask = (labels == O_tag_label).type(torch.int64)
        predictions_mask = (predictions == O_tag_label).type(torch.int64)

        # both of the masks has value 1 when the gold label or predicted label is O
        # we don't want to count cases where both masks has value 1 so we will multiply them
        # finally, these are the samples we DO NOT want to count, so we will flip the result
        O_tag_mask = 1 - (labels_mask * predictions_mask)

        # now O_tag_mask contains all samples where the gold label is not 'O' or the predicated label is not O
        # number of predictions is just a sum of this tensor
        num_predictions = torch.sum(O_tag_mask).item()

        # now check num correct and multiply with mask
        correct_predictions_raw: torch.tensor = (predictions == labels).type(torch.int64)
        correct_predictions: torch.tensor = correct_predictions_raw * O_tag_mask
        num_correct = torch.sum(correct_predictions).item()

        return num_correct, num_predictions


class AcceptorPredictor(BasePredictor):

    def __init__(self, mapper: BaseMapper):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor) -> torch.tensor:
        _, labels_tokens = torch.max(model_outputs, dim=1)

        return labels_tokens


class GreedyLSTMPredictor(BasePredictor):
    def __init__(self, mapper: BaseMapperWithPadding):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor) -> torch.Tensor:
        # dimension of model outputs is batch_size, num_features, sequence_length
        # that is why we are using max on dimension 1 - the features dimension
        _, labels_tokens = torch.max(model_outputs, dim=1)
        return labels_tokens

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> Tuple[int, int]:
        num_correct: int
        num_predictions: int

        self.mapper: BaseMapperWithPadding
        padding_symbol = self.mapper.get_padding_symbol()
        label_padding_index = self.mapper.get_label_idx(padding_symbol)

        # create a mask to distinguish padding from real tokens
        padding_mask = (labels != label_padding_index).type(torch.int64)
        num_predictions = torch.sum(padding_mask).item()

        # compute prediction (greedy, argmax in each time sequence)
        predictions = self.infer_model_outputs(model_outputs)

        # compare between predictions and labels, masking out padding
        correct_predictions_raw = (predictions == labels).type(torch.int64)
        correct_prediction_no_padding = correct_predictions_raw * padding_mask
        num_correct = torch.sum(correct_prediction_no_padding).item()

        return num_correct, num_predictions


class GreedyLSTMPredictorForNER(GreedyLSTMPredictor):

    def __init__(self, mapper: BaseMapperWithPadding):
        super().__init__(mapper)

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> Tuple[int, int]:
        num_correct: int
        num_predictions: int

        # compute prediction (greedy, argmax in each time sequence)
        predictions = self.infer_model_outputs(model_outputs)

        # get the indices of the padding label and the O label
        self.mapper: BaseMapperWithPadding
        padding_symbol = self.mapper.get_padding_symbol()
        label_padding_index = self.mapper.get_label_idx(padding_symbol)
        O_tag_label = self.mapper.get_label_idx('O')

        # create a mask to identify predictions and gold labels of the O tag
        labels_mask = (labels == O_tag_label).type(torch.int64)
        predictions_mask = (predictions == O_tag_label).type(torch.int64)

        # both of the masks has value 1 when the gold label or predicted label is O
        # we don't want to count cases where both masks has value 1 so we will multiply them
        # finally, these are the samples we DO NOT want to count, so we will flip the result
        O_tag_mask = 1 - (labels_mask * predictions_mask)

        # create a mask to distinguish padding from real tokens
        # results in a mask tensor with an entry of 1 where we WANT to count
        padding_mask = (labels != label_padding_index).type(torch.int64)

        # now we want to multiply the prediction mask and the O tag masks
        # stay only with the sample we want to count
        final_mask = O_tag_mask * padding_mask

        # number of predictions is just the number of 1 entries in the mask
        num_predictions = torch.sum(final_mask).item()

        correct_predictions_raw = (predictions == labels).type(torch.int64)
        correct_prediction = correct_predictions_raw * final_mask
        num_correct = torch.sum(correct_prediction).item()

        return num_correct, num_predictions
