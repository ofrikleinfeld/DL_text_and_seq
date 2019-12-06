import torch
import torch.utils.data as data
from mappers import BaseMapper, BEGIN, END, START_LINE


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
                # skip start of file line
                if line.startswith(START_LINE):
                    continue

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
            y = None

        return x, y
