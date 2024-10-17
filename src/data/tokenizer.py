import os
import pandas as pd
from transformers import RobertaTokenizerFast


class Tokenizer():
    def __init__(self):
        # RobertaTokenizerFast is used
        # https://huggingface.co/docs/transformers/en/model_doc/roberta#transformers.RobertaTokenizerFast
        self._tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self._tokenized_df = None
        self._datasetdir = self._find_dataset_dir()
        self._clean_data = self._load_data(self._datasetdir)

    def _find_dataset_dir(self):
        """
        Automatically constructs the dataset directory path based on the current working directory and variable.
        """
        cwd = os.getcwd()  # current working directory
        dataset_dir = os.path.join(cwd, 'data/processed')

        if not os.path.isdir(dataset_dir):
            raise ValueError(f"The directory {dataset_dir} does not exist.")

        if not os.listdir(dataset_dir):
            raise ValueError(f"The directory {dataset_dir} is empty.")
        
        return dataset_dir
    
    def _load_data(self, dataset_dir):
        """
        Loads the dataset from the first CSV file found in the dataset directory.
        """
        # find all CSV files in the directory
        datafiles = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        
        if not datafiles:
            raise ValueError(f"No CSV files found in {dataset_dir}.")
        
        # load the first CSV file found in the directory
        file_path = os.path.join(dataset_dir, datafiles[0])
        print(f"Loading data from: {file_path}")
        dataframe = pd.read_csv(file_path, encoding='utf-8')
        
        return dataframe
    
    def __getitem__(self, index: int):
        """
        Returns a row from the DataFrame at the given index.
        """
        if index >= len(self._clean_data) or index < 0:
            raise IndexError("Index out of range.")
        return self._clean_data.iloc[index]
    
    def __len__(self) -> int:
        return len(self._clean_data)

    def get_df(self):
        return self._clean_data
    
    def _tokenize_dataframe(self):
        """
        Tokenizes each column (except 'bias_rating') of the clean dataframe.
        """
        tokenized_data = {} # dictionary to hold tokenized columns
        max_lengths = self._find_max_token_lengths()

        for column in self._clean_data.columns:
            if column != 'bias_rating':
                # Get the max length for the column (or use a default max length)
                max_length = max_lengths[column]
                print(max_length)

                # tokenize the column
                tokenized_data[f'tokenized_{column}'] = self._clean_data[column].apply(
                    lambda x: self._tokenize_text(x, max_length)['input_ids']
                )

        self._tokenized_df = pd.DataFrame(tokenized_data)
        self._tokenized_df['bias_rating'] = self._clean_data['bias_rating']  # add the 'bias_rating' column back

    def _tokenize_text(self, text, max_length=None):
        """
        Tokenize the text using the BERT tokenizer. If max_length is provided,
        tokenize with padding and truncation. Otherwise, return the unpadded and untruncated tokens.
        """
        if max_length:
            # version of tokenizer with truncation and padding to the specified max_length
            tokenized_output = self._tokenizer(
                text,
                max_length=max_length,
                padding='max_length', 
                truncation=True,
                return_tensors='pt'
            )
        else:
            # version of tokenizer without truncation or padding
            # to find the true token length
            tokenized_output = self._tokenizer(
                text,
                padding=False,
                truncation=False,
                return_tensors='pt'
            )
        return tokenized_output

    def _find_max_token_lengths(self):
        """
        Finds the maximum token lengths for each column in the DataFrame.
        """
        max_lengths = {}

        for column in self._clean_data.columns:
            if column != 'bias_rating':  # skip 'bias_rating'
                # tokenize each sentance in a column without applying max_length
                max_length = self._clean_data[column].apply(
                    lambda x: len(self._tokenize_text(x)['input_ids'][0])
                ).max()

                max_lengths[column] = max_length

        return max_lengths
    
    def tokenize(self):
        self._tokenize_dataframe()


if __name__ == "__main__":
    pass
