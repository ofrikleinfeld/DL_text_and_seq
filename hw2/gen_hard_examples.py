import argparse
import sys
from typing import List, Callable
from random import shuffle, randint, sample

POSITIVE_PATTERN = 'POS'
NEGATIVE_PATTERN = 'NEG'


def generate_positive_example_equal():
    randnum = randint(1, 20)
    s = ('a' * randnum) + ('b' * randnum)
    s = ''.join(sample(s, len(s)))
    return s

def generate_negative_example_equal():
    randnum_1 = randnum_2 = 0
    while randnum_1 == randnum_2:
        randnum_1 = randint(1, 20)
        randnum_2 = randint(1, 20)
    s = ('a' * randnum_1) + ('b' * randnum_2)
    s = ''.join(sample(s, len(s)))
    return s

def generate_examples_equal(pattern: str, n: int) -> List[str]:
    examples = set()
    if pattern == POSITIVE_PATTERN:
        generator = generate_positive_example_equal
    else:
        generator = generate_negative_example_equal
    while len(examples) != n:
        s = generator()
        examples.add(s)
    return list(examples)


def isPrime(n):
    # Corner cases
    if (n <= 1):
        return False
    if (n <= 3):
        return True
    if (n % 2 == 0 or n % 3 == 0):
        return False
    i = 5
    while (i * i <= n):
        if (n % i == 0 or n % (i + 2) == 0):
            return False
        i = i + 6
    return True

def generate_positive_example_prime():
    D = {}
    q = 2
    while True:
        if q not in D:
            yield 'a' * q
            D[q * q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        q += 1

def generate_negative_example_prime():
    while True:
        n = randint(1,10000)
        if not isPrime(n):
            yield 'a' * n

def generate_examples_prime(pattern: str, n: int) -> List[str]:
    examples = set()
    if pattern == POSITIVE_PATTERN:
        generator = generate_positive_example_prime()
    else:
        generator = generate_negative_example_prime()
    while len(examples) != n:
        s = next(generator)
        examples.add(s)
    return list(examples)



def generate_positive_example_palindrome():
   n = randint(1,20)
   left = ''
   right = ''
   for i in range(n):
       r = randint(0,1)
       if r == 0:
           left += 'a'
           right = 'a' + right
       else:
           left += 'b'
           right = 'b' + right
   if randint(0,1) == 0:
        left += 'c'
   return (left +  right)


def generate_negative_example_palindrome():
   n = randint(1,41)
   s = ''
   for i in range(n):
       r = randint(1,2)
       if r == 1:
           s += 'a'
       elif r == 2:
           s += 'b'
   r = randint(0,1)
   if r == 0:
       rand_idx = sample(list(range(len(s))), 1)[0]
       s = s[:rand_idx] + 'c' + s[rand_idx:]
   return s

def generate_examples_palindrome(pattern: str, n: int) -> List[str]:
    examples = set()
    if pattern == POSITIVE_PATTERN:
        generator = generate_positive_example_palindrome
    else:
        generator = generate_negative_example_palindrome
    while len(examples) != n:
        s = generator()
        examples.add(s)
    return list(examples)


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


def generate_training_test_sets(num_training: int, train_out_path: str, num_test: int, test_out_path: str, generate_examples: Callable):
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
    generate_examples_parser.add_argument("--generator_type", type=str, required=False, choices=["equal", "prime", "palindrome"],
                                          help="which type of hard examples to generate")
    # create the parser for the generate datasets command
    generate_datasets_parser = subparsers.add_parser('train_test')
    generate_datasets_parser.add_argument("--num_training", type=int, required=True, metavar="500", help="number of training examples to generate")
    generate_datasets_parser.add_argument("--train_output_path", type=str, required=False, help="path to a file to save generated training data")
    generate_datasets_parser.add_argument("--num_test", type=int, required=True, metavar="500", help="number of training examples to generate")
    generate_datasets_parser.add_argument("--test_output_path", type=str, required=False, help="path to a file to save generated training data")
    generate_datasets_parser.add_argument("--generator_type", type=str, required=False, choices = ["equal", "prime", "palindrome"],
                                          help="which type of hard examples to generate")

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
        generator_type = args.generator_type
        if generator_type == 'equal':
            generate_examples = generate_examples_equal
        elif generator_type == 'prime':
            generate_examples = generate_examples_prime
        elif generator_type == 'palindrome':
            generate_examples = generate_examples_palindrome
        generated_examples = generate_examples(type_pattern, num_examples)
        write_examples_to_file(generated_examples, examples_path)

    if mode == "train_test":
        num_training = args.num_training
        train_output_path = args.train_output_path
        num_test = args.num_test
        test_output_path = args.test_output_path
        generator_type = args.generator_type
        if generator_type == 'equal':
            generate_examples = generate_examples_equal
        elif generator_type == 'prime':
            generate_examples = generate_examples_prime
        elif generator_type == 'palindrome':
            generate_examples = generate_examples_palindrome

        generate_training_test_sets(num_training, train_output_path, num_test, test_output_path, generate_examples)
