import argparse
import sys
from typing import List
from random import shuffle

import rstr

POSITIVE_PATTERN = r"[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+"
NEGATIVE_PATTERN = r"[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+"


def generate_examples(pattern: str, n: int) -> List[str]:
    examples = []
    for i in range(n):
        examples.append(rstr.xeger(pattern))

    return examples


def write_examples_to_file(examples: List[str], output_path: str, labels: List[int] = None) -> None:
    with open(output_path, "w") as f:
        if labels is not None:
            # shuffle examples
            examples_with_labels = list(zip(examples, labels))
            shuffle(examples_with_labels)
            for example, label in examples_with_labels:
                f.write(f"{example}\t{label}\n")
        else:
            for example in examples:
                f.write(f"{example}\n")


def generate_training_test_sets(num_training: int, train_out_path: str, num_test: int, test_out_path: str):
    # number of examples to generate
    num_positive_train = num_training // 2
    num_negative_train = num_training - num_positive_train
    num_positive_test = num_test // 2
    num_negative_test = num_test - num_positive_test

    # generate samples
    positive_train = generate_examples(POSITIVE_PATTERN, num_positive_train)
    negative_train = generate_examples(NEGATIVE_PATTERN, num_negative_train)
    positive_test = generate_examples(POSITIVE_PATTERN, num_positive_test)
    negative_test = generate_examples(NEGATIVE_PATTERN, num_negative_test)

    # generate labels
    positive_train_labels = [1] * num_positive_train
    negative_train_labels = [0] * num_negative_train
    positive_test_labels = [1] * num_positive_test
    negative_test_labels = [0] * num_negative_train

    train_samples = positive_train + negative_train
    train_labels = positive_train_labels + negative_train_labels
    test_samples = positive_test + negative_test
    test_labels = positive_test_labels + negative_test_labels

    # write to output files
    write_examples_to_file(train_samples, train_out_path, train_labels)
    write_examples_to_file(test_samples, test_out_path, test_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generating examples for RNN Acceptor')
    subparsers = parser.add_subparsers()

    # create the parser for the generate examples command
    generate_examples_parser = subparsers.add_parser('examples')
    generate_examples_parser.add_argument("--type", type=str, required=True, metavar="positive",choices=["positive", "negative"], help="generate example from a specific type")
    generate_examples_parser.add_argument("--n", type=int, required=True, metavar="500", help="number of examples to generate")
    generate_examples_parser.add_argument("--output_path", type=str, required=True, help="path to a file to save results to")

    # create the parser for the generate datasets command
    generate_datasets_parser = subparsers.add_parser('train_test')
    generate_datasets_parser.add_argument("--num_training", type=int, required=True, metavar="500", help="number of training examples to generate")
    generate_datasets_parser.add_argument("--train_output_path", type=str, required=False, help="path to a file to save generated training data")
    generate_datasets_parser.add_argument("--num_test", type=int, required=True, metavar="500", help="number of training examples to generate")
    generate_datasets_parser.add_argument("--test_output_path", type=str, required=False, help="path to a file to save generated training data")

    # parse the arguments
    args = parser.parse_args(sys.argv[1:])
    mode = sys.argv[1]

    if mode == "examples":
        label_type = args.type
        num_examples = args.n
        examples_path = args.output_path

        if label_type == "positive":
            type_pattern = POSITIVE_PATTERN
        else:
            type_pattern = NEGATIVE_PATTERN

        generated_examples = generate_examples(type_pattern, num_examples)
        write_examples_to_file(generated_examples, examples_path)

    if mode == "train_test":
        num_training = args.num_training
        train_output_path = args.train_output_path
        num_test = args.num_test
        test_output_path = args.test_output_path

        generate_training_test_sets(num_training, train_output_path, num_test, test_output_path)
