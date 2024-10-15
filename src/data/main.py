from data_loader import DataSetLoader
from data_preprocessing import TextPreprocessor
from tokenizer import Tokenizer


def main():
    preprocessed_dataset = TextPreprocessor()
    preprocessed_dataset.preprocess()
    data = preprocessed_dataset.get_df()

    loader = DataSetLoader()
    loader(data, "data/processed/clean_data.csv")

    tokenized_dataset = Tokenizer()
    tokenized_dataset.tokenize()
    print(len(tokenized_dataset))

    loader = DataSetLoader()
    loader(tokenized_dataset, "data/tokenized/tokenized_clean_data.csv")


if __name__ == "__main__":
    main()
