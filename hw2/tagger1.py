from mappers import TokenMapperUnkCategory
from datasets import WindowDatasetUnkCategories

if __name__ == '__main__':
    train_path = "en_ewt-ud-train.conllu"
    mapper = TokenMapperUnkCategory(min_frequency=30)
    mapper.create_mapping(train_path)
    dataset = WindowDatasetUnkCategories(train_path, mapper)

    print(len(dataset))
    sample, label = dataset[3]
    print(sample)
    print(label)
