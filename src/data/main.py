from data_loader import DataSetLoader
from data_preprocessing import TextPreprocessor
import pandas as pd


def main():
    preprocessed_dataset = TextPreprocessor()
    preprocessed_dataset.preprocess()
    data = preprocessed_dataset.get_df()

    loader = DataSetLoader()
    loader(data, "data/processed/clean_data.csv")


if __name__ == "__main__":
    main()
