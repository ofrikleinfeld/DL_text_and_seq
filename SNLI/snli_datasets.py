from typing import Tuple, List

import torch

from pos_and_ner.datasets import BaseDataset
from SNLI.snli_mappers import SNLIMapperWithGloveIndices


class SNLIDataset(BaseDataset):
    def __init__(self, filepath: str, mapper: SNLIMapperWithGloveIndices, sequence_length: int = 25):
        super().__init__(filepath, mapper)
        self.sequence_length = sequence_length

    def _init_dataset(self) -> None:
        with open(self.filepath, "r", encoding="utf8") as f:
            header_line = True
            for line in f:
                # skip header line, first line in the file
                if header_line:
                    header_line = False
                    continue

                line_tokens = line[:-1].split(self.mapper.split_char)  # remove end of line
                label = line_tokens[0]
                if label != self.mapper.unknown_label_symbol:

                    sentence_1 = line_tokens[5].split(" ")
                    sentence_2 = line_tokens[6].split(" ")
                    sentence_1 = self._prune_or_pad_sample(sentence_1)
                    sentence_2 = self._prune_or_pad_sample(sentence_2)

                    self.samples.append((sentence_1, sentence_2))
                    self.labels.append(label)

    def _prune_or_pad_sample(self, sample: List[str]) -> List[str]:
        # padding or pruning
        self.mapper: SNLIMapperWithGloveIndices
        const_len_sample: List[str]
        sample_length = len(sample)

        if sample_length > self.sequence_length:
            const_len_sample = sample[:self.sequence_length]
        else:
            padding_length = self.sequence_length - sample_length
            const_len_sample = sample + [self.mapper.get_padding_symbol()] * padding_length

        return const_len_sample

    def __getitem__(self, item_idx: int) -> Tuple[torch.tensor, torch.tensor]:
        self.init_dataset_if_not_initiated()

        # check if we have labels or it is a blind test set
        if len(self.labels) > 0:
            label = self.labels[item_idx]
            label_index = self.mapper.get_label_idx(label)
            y = torch.tensor(label_index)
        else:
            y = torch.tensor([])

        # even for test set we anyway have samples to predict
        sample = self.samples[item_idx]
        sentence_1, sentence_2 = sample
        sentence_1_indices = [self.mapper.get_token_idx(word) for word in sentence_1]
        sentence_2_indices = [self.mapper.get_token_idx(word) for word in sentence_2]
        sentence_1_tensor = torch.tensor(sentence_1_indices)
        sentence_2_tensor = torch.tensor(sentence_2_indices)

        x = torch.stack([sentence_1_tensor, sentence_2_tensor], dim=1)

        return x, y
