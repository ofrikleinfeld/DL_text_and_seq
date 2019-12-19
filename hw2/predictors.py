from typing import List, Tuple

import torch

from mappers import BaseMapper, BaseMapperWithPadding


class BasePredictor(object):

    def __init__(self, mapper: BaseMapper):
        self.mapper = mapper

    def infer_model_outputs(self, model_outputs: torch.tensor):
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
        num_correct = 0
        num_predictions = 0
        gold_labels = []
        for label_idx in labels:
            gold_labels.append(label_idx.item())

        predictions = self.infer_model_outputs(model_outputs)
        for i in range(len(predictions)):
            current_prediction = predictions[i]
            current_label = gold_labels[i]
            num_predictions += 1

            if current_prediction == current_label:
                num_correct += 1

        return num_correct, num_predictions


class WindowModelPredictor(BasePredictor):

    def __init__(self, mapper: BaseMapper):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor):
        predictions = []
        _, labels_tokens = torch.max(model_outputs, dim=1)

        for i in range(len(model_outputs)):  # every sample in case of batch (even batch of size 1)
            current_prediction = labels_tokens[i].item()
            predictions.append(current_prediction)

        return predictions


class WindowNERTaggerPredictor(WindowModelPredictor):

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> (int, int):
        # special case of unbalanced labels
        # don't compute accuracy over trivial cases of the 'O' tag
        num_correct = 0
        num_predictions = 0
        gold_labels = []
        for label_idx in labels:
            gold_labels.append(label_idx.item())

        predictions = self.infer_model_outputs(model_outputs)
        for i in range(len(predictions)):
            current_prediction = predictions[i]
            current_label = gold_labels[i]

            # don't count cases where predicated label and gold label are 'O'
            # get the index of the label 'O'
            outside_index = self.mapper.get_label_idx('O')
            if not (current_label == outside_index and current_prediction == outside_index):

                if current_prediction == current_label:
                    num_correct += 1

                num_predictions += 1

        return num_correct, num_predictions


class AcceptorPredictor(BasePredictor):

    def __init__(self, mapper: BaseMapper):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor) -> List[int]:
        predictions = []
        _, labels_tokens = torch.max(model_outputs, dim=1)

        for i in range(len(model_outputs)):  # every sample in case of batch (even batch of size 1)
            current_prediction = labels_tokens[i].item()
            predictions.append(current_prediction)

        return predictions


class GreedyLSTMPredictor(BasePredictor):
    def __init__(self, mapper: BaseMapperWithPadding):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor) -> List[List[int]]:
        batch_predictions = []
        single_sentence_prediction = []
        batch_size, sequence_length = model_outputs.size()

        _, labels_tokens = torch.max(model_outputs, dim=2)
        for i in range(batch_size):  # number of samples in batch
            for j in range(sequence_length):  # should be length of maximum sequence in dataset
                current_prediction = labels_tokens[i][j].item()
                single_sentence_prediction.append(current_prediction)

            batch_predictions.append(single_sentence_prediction)
            single_sentence_prediction = []

        return batch_predictions

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> Tuple[int, int]:
        num_correct = 0
        num_predictions = 0
        self.mapper: BaseMapperWithPadding
        padding_index = self.mapper.get_padding_index()

        batch_size, sequence_length = labels.size()
        gold_labels = []
        sequence_gold_labels = []
        for i in range(batch_size):
            for j in range(sequence_length):
                label_index = labels[i][j].item()
                if label_index != padding_index:
                    sequence_gold_labels.append(label_index)

            gold_labels.append(sequence_gold_labels)
            sequence_gold_labels = []

        predictions = self.infer_model_outputs(model_outputs)
        for i in range(len(gold_labels)):
            sequence_labels = gold_labels[i]
            for j in range(len(sequence_labels)):
                current_prediction = predictions[i][j]
                current_label = sequence_labels[j]

                if current_prediction == current_label:
                    num_correct += 1

                num_predictions += 1

        return num_correct, num_predictions
