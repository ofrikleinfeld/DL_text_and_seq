import torch
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

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> (int, int):
        return NotImplementedError("A class deriving from BasePredictor must implement infer_model_outputs_with_gold_labels method")


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

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> (int, int):
        num_correct = 0
        num_predictions = 0
        gold_labels = []
        for label_idx in labels:
            gold_labels.append(self.mapper.get_label_idx(label_idx))

        predictions = self.infer_model_outputs(model_outputs)
        for i in range(len(predictions)):
            current_prediction = predictions[i]
            current_label = gold_labels[i]
            num_predictions += 1

            if current_prediction == current_label:
                num_correct += 1

        return num_correct, num_predictions


class WindowNERTaggerPredictor(WindowModelPredictor):

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> (int, int):
        # special case of unbalanced labels
        # don't compute accuracy over trivial cases of the 'O' tag
        num_correct = 0
        num_predictions = 0
        gold_labels = []
        for label_idx in labels:
            gold_labels.append(self.mapper.get_label_idx(label_idx))

        predictions = self.infer_model_outputs(model_outputs)
        for i in range(len(predictions)):
            current_prediction = predictions[i]
            current_label = gold_labels[i]

            # don't count cases where predicated label and gold label are 'O'
            if not (current_label == 'O' and current_prediction == 'O'):

                if current_prediction == current_label:
                    num_correct += 1

                num_predictions += 1

        return num_correct, num_predictions
