from mappers import TokenMapperUnkCategory
from datasets import WindowDatasetUnkCategories

if __name__ == '__main__':
    train_path = "pos/train"
    test_path = "pos/test"
    mapper = TokenMapperUnkCategory(min_frequency=5, split_char=" ")
    mapper.create_mapping(train_path)
    dataset = WindowDatasetUnkCategories(test_path, mapper)

    print(len(dataset))
    sample, label = dataset[3]
    print(sample)
    print(label)
