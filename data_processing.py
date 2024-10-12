
import pandas as pd
import os
from transformers import BertTokenizer, TFBertModel

MAX_TOKEN_COUNT = 512

# BERT tokenizer is used 
# https://huggingface.co/google-bert/bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

"""
This function takes one csv file and returns another csv file
with selected feature columns
"""
def make_processed_csv(path):

    df = pd.read_csv(path, index_col = 0)

    df['h_text'] = df['heading'] + ' ' + df['text']

    df_p = df[['h_text', 'bias_rating']]

    df_p['h_text'] = df_p['h_text'].astype(str) # Doing this because otherwise it doesn't work

    df_p.to_csv(os.path.join(os.path.dirname(path), "processed_data.csv"), index=False)
    return df_p


"""
Tokenizes the given sentence.
"""
def tokenize(string, max_length = MAX_TOKEN_COUNT, padding = 'max_length', return_tensors = 'tf'):
    return tokenizer(string, padding = padding, return_tensors = return_tensors)


"""
Finds the sentence with the maximum number of tokens in the df
"""
def max_sentence(dataframe):

    max_sentence, max_token_count = max(
    ((sentence, len(tokenize(sentence)['input_ids'][0])) for sentence in dataframe),
    key=lambda x: x[1])

    return max_sentence, max_token_count
