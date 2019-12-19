from typing import Tuple, List

import torch
import torch.utils.data as data

from mappers import BaseMapper, BaseMapperWithPadding, TokenMapperWithSubWords, BEGIN, END


class WindowDataset(data.Dataset):
    """
    Pytorch's Dataset derived class to create data sample from
    a path to a file
    """
    def __init__(self, filepath: str, mapper: BaseMapper, window_size: int = 2):
        super().__init__()
        self.filepath = filepath
        self.mapper = mapper
        self.window_size = window_size
        self.samples = []
        self.labels = []

    def _load_file(self) -> None:
        curr_sent = []
        curr_labels = []
        with open(self.filepath, "r", encoding="utf8") as f:
            for line in f:
                if line == "\n":  # marks end of sentence
                    self._create_window_samples_from_sent(curr_sent, curr_labels)
                    # clear before reading next sentence
                    curr_sent = []
                    curr_labels = []

                else:  # line for word in a sentence
                    tokens = line[:-1].split(self.mapper.split_char)

                    # verify if it is a dataset with or without labels
                    if len(tokens) == 2:  # train dataset
                        word, label = tokens[0], tokens[1]
                        curr_labels.append(label)
                    else:
                        word = tokens[0]

                    # anyway we will have a word to predict
                    curr_sent.append(word)

    def _create_window_samples_from_sent(self, sent, labels) -> None:
        sentence_length = len(sent)
        sent = [BEGIN] * self.window_size + sent + [END] * self.window_size

        # iterate for sentence_length times
        # start with index of real word
        for i in range(self.window_size, sentence_length + self.window_size):
            current_sample = sent[i - self.window_size:i + 1 + self.window_size]
            self.samples.append(current_sample)

        for label in labels:
            self.labels.append(label)

    def __len__(self) -> int:
        # perform lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        return len(self.samples)

    def __getitem__(self, item_idx: int) -> (torch.tensor, torch.tensor):
        # lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        # retrieve sample and transform from tokens to indices
        sample = self.samples[item_idx]
        sample_indices = [self.mapper.get_token_idx(word) for word in sample]
        x = torch.tensor(sample_indices)

        # verify if it a train/dev dataset of a test dataset
        if len(self.labels) > 0:  # we have labels
            label = self.labels[item_idx]
            label_index = self.mapper.get_label_idx(label)
            y = torch.tensor(label_index)
        else:
            y = torch.tensor([])

        return x, y


class WindowWithSubWordsDataset(WindowDataset):
    def __init__(self, filepath: str, mapper: TokenMapperWithSubWords, window_size: int = 2):
        super().__init__(filepath, mapper, window_size)
        self.prefixes = []
        self.suffixes = []

    def _create_window_samples_for_prefix_suffix(self, prefixes, suffixes):
        sentence_length = len(prefixes)
        prefixes = [BEGIN] * self.window_size + prefixes + [END] * self.window_size
        suffixes = [BEGIN] * self.window_size + suffixes + [END] * self.window_size

        # iterate for sentence_length times
        # start with index of real word
        for i in range(self.window_size, sentence_length + self.window_size):
            current_prefix = prefixes[i - self.window_size:i + 1 + self.window_size]
            current_suffix = suffixes[i - self.window_size:i + 1 + self.window_size]
            self.prefixes.append(current_prefix)
            self.suffixes.append(current_suffix)

    def _load_file(self) -> None:
        curr_sent = []
        curr_labels = []
        curr_prefixes = []
        curr_suffixes = []

        with open(self.filepath, "r", encoding="utf8") as f:
            for line in f:
                if line == "\n":  # marks end of sentence
                    self._create_window_samples_from_sent(curr_sent, curr_labels)
                    self._create_window_samples_for_prefix_suffix(curr_prefixes, curr_suffixes)
                    # clear before reading next sentence
                    curr_sent = []
                    curr_labels = []
                    curr_prefixes = []
                    curr_suffixes = []

                else:  # line for word in a sentence
                    tokens = line[:-1].split(self.mapper.split_char)

                    # verify if it is a dataset with or without labels
                    if len(tokens) == 2:  # train dataset
                        word, label = tokens[0], tokens[1]
                        curr_labels.append(label)
                    else:
                        word = tokens[0]

                    # anyway we will have a word to predict
                    prefix = word[:3]
                    suffix = word[-3:]
                    curr_prefixes.append(prefix)
                    curr_suffixes.append(suffix)
                    curr_sent.append(word)

    def __getitem__(self, item_idx: int) -> (tuple, torch.tensor):
        if len(self.samples) == 0:
            self._load_file()

        # retrieve sample and transform from tokens to indices
        sample = self.samples[item_idx]
        prefixs = self.prefixes[item_idx]
        suffixes = self.suffixes[item_idx]
        sample_indices = [self.mapper.get_token_idx(word) for word in sample]
        prefixes_indices = [self.mapper.get_prefix_index(prefix) for prefix in prefixs]
        suffixes_indices = [self.mapper.get_suffix_index(suffix) for suffix in suffixes]

        sample_indices = torch.tensor(sample_indices)
        prefixes_indices = torch.tensor(prefixes_indices)
        suffixes_indices = torch.tensor(suffixes_indices)
        x = torch.stack([sample_indices, prefixes_indices, suffixes_indices])

        # verify if it a train/dev dataset of a test dataset
        if len(self.labels) > 0:  # we have labels
            label = self.labels[item_idx]
            label_index = self.mapper.get_label_idx(label)
            y = torch.tensor(label_index)
        else:
            y = torch.tensor([])

        return x, y


class RegularLanguageDataset(data.Dataset):

    def __init__(self, filepath: str, mapper: BaseMapperWithPadding, sequence_length: int = 65):
        self.filepath = filepath
        self.mapper = mapper
        self.sequence_length = sequence_length
        self.samples = []
        self.labels = []

    def _load_file(self) -> None:
        with open(self.filepath, "r", encoding="utf8") as f:
            for line in f:
                sample, label = line[:-1].split(self.mapper.split_char)
                sample = self._prune_or_pad_sample(sample)
                self.samples.append(sample)
                self.labels.append(label)

    def _prune_or_pad_sample(self, sample: str) -> str:
        # padding or pruning
        const_len_sample: str
        sample_length = len(sample)

        if sample_length > self.sequence_length:
            const_len_sample = sample[:self.sequence_length]

        else:
            padding_length = self.sequence_length - sample_length
            const_len_sample = sample + self.mapper.get_padding_symbol() * padding_length

        return const_len_sample

    def __len__(self) -> int:
        # perform lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        return len(self.samples)

    def __getitem__(self, item_idx: int) -> Tuple[torch.tensor, torch.tensor]:
        # lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        # retrieve sample and transform from tokens to indices
        sample = self.samples[item_idx]
        label = self.labels[item_idx]

        sample_indices = [self.mapper.get_token_idx(word) for word in sample]
        label_index = self.mapper.get_label_idx(label)

        x = torch.tensor(sample_indices)
        y = torch.tensor(label_index)

        return x, y


class BiLSTMDataset(data.Dataset):
    def __init__(self, filepath: str, mapper: BaseMapperWithPadding, sequence_length: int = 65):
        self.filepath = filepath
        self.mapper = mapper
        self.sequence_length = sequence_length
        self.samples = []
        self.labels = []

    def _load_file(self) -> None:
        with open(self.filepath, "r", encoding="utf8") as f:
            curr_sentence = []
            curr_labels = []
            for line in f:

                if line == "\n":  # empty line denotes end of a sentence
                    curr_sentence = self._prune_or_pad_sample(curr_sentence)
                    curr_labels = self._prune_or_pad_sample(curr_labels)

                    self.samples.append(curr_sentence)
                    self.labels.append(curr_labels)
                    curr_sentence = []
                    curr_labels = []

                else:  # append word and label to current sentence
                    word, label = line[:-1].split(self.mapper.split_char)
                    curr_sentence.append(word)
                    curr_labels.append(label)

    def _prune_or_pad_sample(self, sample: List[str]) -> List[str]:
        # padding or pruning
        const_len_sample: List[str]
        sample_length = len(sample)

        if sample_length > self.sequence_length:
            const_len_sample = sample[:self.sequence_length]
        else:
            padding_length = self.sequence_length - sample_length
            const_len_sample = sample + [self.mapper.get_padding_symbol()] * padding_length

        return const_len_sample
