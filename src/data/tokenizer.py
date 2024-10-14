import numpy as np
from transformers import BertTokenizer


MAX_TOKEN_COUNT = 512

class Tokenizer():
    def __init__(self):
        # BERT tokenizer is used
        # https://huggingface.co/google-bert/bert-base-uncased
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _tokenize_text(self, text):
        """
        Tokenize the text using the BERT tokenizer and return input IDs and attention masks.
        """
        # Tokenize the text with truncation and padding
        tokenized_output = self.tokenizer(
            text,
            max_length=MAX_TOKEN_COUNT,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        return tokenized_output

    def _tokenize(self, string, max_length=MAX_TOKEN_COUNT, padding='max_length', return_tensors='np'):
        """
        Tokenizes a string using the BERT tokenizer with specified settings for padding and truncation.
        """
        return self.tokenizer(string, padding=padding, truncation=True, max_length=max_length, return_tensors=return_tensors)

    
    def max_sentence(self, dataframe):
        """
        Finds the sentence with the maximum number of tokens in the dataframe.
        
        Args:
            dataframe (pd.DataFrame): A DataFrame containing sentences.

        Returns:
            tuple: A tuple containing the sentence with the most tokens and its token count.
        """
        # Tokenize each sentence and count the number of tokens
        max_sentence, max_token_count = max(
            ((sentence, len(self._tokenize(sentence)['input_ids'][0])) for sentence in dataframe['text']),
            key=lambda x: x[1]
        )
        return max_sentence, max_token_count
