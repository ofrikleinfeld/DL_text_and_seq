from mappers import TokenMapperUnkCategory
from datasets import WindowDataset

if __name__ == '__main__':
    train_path = "pos/train"
    test_path = "pos/test"
    mapper = TokenMapperUnkCategory(min_frequency=5, split_char=" ")
    mapper.create_mapping(train_path)
    dataset = WindowDataset(test_path, mapper)

    print(len(dataset))
    for i in range(len(dataset)):
        sample, label = dataset[i]
        print(sample)
        print(label)
