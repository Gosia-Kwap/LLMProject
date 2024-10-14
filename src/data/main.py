import os
import pandas as pd
import unicodedata
import ftfy
from data_loader import DataSetLoader
from data_preprocessing import TextPreprocessor
from tokenizer import Tokenizer


def main():
    print("Welcome to data preprocessing pipeline!")
    preprocessed_dataset = TextPreprocessor()
    preprocessed_dataset.preprocess()
    data = preprocessed_dataset.get_df()

    tokenizer = Tokenizer()

    loader = DataSetLoader()
    loader(data, "data/processed/clean_data.csv")

    # preprocessor.find_non_ascii_in_dataframe(columns=['title', 'text'])

    # loader = DataLoader(preprocessed_data)

    # /Users/Administrator/Documents/University/Year 3/LLMs/LLMProject/data/raw.csv

#     import unicodedata

# # Normalize to NFC form (composed form)
# normalized_text = unicodedata.normalize('NFC', text)

if __name__ == "__main__":
    main()
