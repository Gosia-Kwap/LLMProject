import os
import pandas as pd
import unicodedata
import ftfy


class TextPreprocessor():
    def __init__(self):
        """
        Initializes the preprocessing class by automatically finding the data path and loading the data.
        """
        self._datasetdir = self._find_dataset_dir()
        self._dataframe = self._load_data(self._datasetdir)

    def _find_dataset_dir(self):
        """
        Automatically constructs the dataset directory path based on the current working directory and variable.
        """
        cwd = os.getcwd()  # current working directory
        dataset_dir = os.path.join(cwd, 'data/raw')

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

    def _extract_columns(self):
        """
        This function takes one csv file and returns another csv file
        with selected feature columns
        """
        columns_to_keep = ['heading', 'text', 'bias_rating']
        self._dataframe = self._dataframe[columns_to_keep]

    def _combine_columns(self):
        self._dataframe['heading_text'] = self._dataframe['heading'] + ' ' + self._dataframe['text']
        self._dataframe = self._dataframe[['heading_text', 'bias_rating']]
        self._dataframe['heading_text'] = self._dataframe['heading_text'].astype(str)
    
    def _check_missing_values(self):
        """
        Finds and deletes rows with missing values.
        """
        missing_values = self._dataframe.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found.")
        else:
            print(f"Missing values are found: \n{missing_values}.\nRows with NA values are removed.")
            self._dataframe = self._dataframe.dropna()

    def _check_duplicates(self):
        """
        Finds duplicates. First occurance of data point is kept,
        further repetitions are marked as duplicates and deleted.
        """
        duplicates = self._dataframe.duplicated(subset=['text'])

        if duplicates.any():
            count = 0
            for i in duplicates:
                if i:
                    count += 1
            print(f"There are found {count} duplicates.")
            self._dataframe = self._dataframe.drop_duplicates() # remove duplicates
            print("Duplicates are successfully removed.")
        else:
            print("No duplicate rows are found.")

    def _find_and_replace_non_ascii(self, columns):
        """
        Applies character replacemnt for column/s
        """
        for column in columns:
            idx, _, sentence_set = self._find_non_ascii(column)
            if len(idx) != 0:
                for idx, sentence in zip(idx, sentence_set):
                    self._dataframe.at[idx, column] = self._replace_non_ascii(sentence)

    def _replace_non_ascii(self, sentence):
        '''
        Replaces non-ASCII characters to valid ones in a specific sentance.
        '''
        replacements = {
        '“': '"', '”': '"', '″': '"', '’': "'", '‘': "'", '—': '-',
        '─': '-', '–': '-', 'ã': 'a', '‚': ',', '€': 'EUR', '…': '...',
        '½': '1/2', 'ƒ': 'f', 'Â': 'A', 'é': 'e', 'è': 'e', 'ê': 'e',
        'ë': 'e', 'É': 'E', 'ó': 'o', 'ö': 'o', 'ñ': 'n', 'ü': 'u',
        'ú': 'u', '£': 'GBP', '―': '-', 'ğ': 'g', 'á': 'a', 'à': 'a',
        '❤': '<3', '§': 'section', '′':'`', '´': '`', 'í':'i', 'î': 'i',
        'ı': 'i', 'ï': 'i', 'š': 's', 'ş': 's'
        }
        for non_ascii_char, ascii_char in replacements.items():
            sentence = sentence.replace(non_ascii_char, ascii_char)
        return sentence
    
    def _find_non_ascii(self, column):
        """
        Checks each sentence in the specified column for any non-ASCII
        characters and saves their positions and corresponding sentences.
        """
        non_ascii_chars = {}
        indices = []
        sentence_set = []
        for idx, sentence in self._dataframe[column].items():
            for char in sentence:
                if ord(char) > 127:
                    indices.append(idx)
                    sentence_set.append(sentence)
                    non_ascii_chars[char] = ord(char)
        return indices, non_ascii_chars, sentence_set
    
    def _show_ambiguous_unicode(self, columns):
        '''
        If any non-ASCII characters are found, it prints the row index, the character itself,
        its Unicode code point, and its Unicode name.
        '''
        for column in columns:
            idxs, non_ascii_chars, _ = self._find_non_ascii(column)
            if len(idxs) != 0:
                for idx in idxs:
                    print(f"\nRow {idx} in column '{column}' contains non-ASCII characters:")
                    for char, code_point in non_ascii_chars.items():
                        print(f"Character: {char} | Unicode code point: {code_point} | Unicode name: {unicodedata.name(char, 'Unknown')}")
        if len(idxs) == 0:
            print("Text does not contain any ambiguous characters.")

    def remove_specified_characters(self, text, chars_to_remove):
        # remove and replace with an empty string
        for char in chars_to_remove:
            text = text.replace(char, '')
        return text
    
    def remove_specified_characters_from_dataframe(self):
        chars_to_remove = ['©', '°', '™', 'ç', '■', '\x9d', '\xad', '\u2009', '蔡', '英', '文', '吳', '釗', '燮', '•', '¿', '\u202f', '►', '\u200c', '✔', 'ø', '️', '¬', '•']
        # apply the replacement
        for column in ['heading_text']:
            self._dataframe[column] = self._dataframe[column].apply(lambda x: self.remove_specified_characters(x, chars_to_remove))


    def _fix_text_encoding(self, columns):
        """
        Corrects garbled or misencoded characters that arise from misinterpreted text encodings
        (for example UTF-8).
        """    
        # Iterate over the specified columns
        for column in columns:
            for idx, value in self._dataframe[column].items():
                # Apply the text encoding fix to each value in the column
                fixed_value = ftfy.fix_text(value)
                # Update the dataframe with the fixed value
                self._dataframe.at[idx, column] = fixed_value

    def _to_lowercase(self, columns):
        for column in columns:
            original_column = self._dataframe[column]
            converted_values = [] # list to store the converted values
            
            # iterate over each sentence in the column
            for sentence in original_column:
                if isinstance(sentence, str):
                    converted_values.append(sentence.lower())
                else:
                    converted_values.append(sentence)

            self._dataframe[column] = converted_values # update column with new lowercase sentances

    def _delete_extra_whitespace(self, columns):
        for column in columns:
            original_length = len(self._dataframe[column])
            for idx, sentence in self._dataframe.iterrows():
            # Check if there is leading/trailing whitespace or extra spaces between words
                if isinstance(sentence, str) and self._detect_whitespaces(sentence):
                    cleaned_sentence = ' '.join(sentence.split())
                    self._dataframe.at[idx, column] = cleaned_sentence # Update the dataframe with the cleaned sentence

    def _detect_whitespaces(self, sentence):
        return sentence != sentence.strip() or ' '.join(sentence.split()) != sentence

    def _normalize_text(self):
        """
        Normalize text by lowercasing, expanding contractions, removing special characters, 
        and removing stop words (if desired).
        """
        columns=['heading_text']

        # ------- Ambiguous Unicode Characters ---------
        # Next steps take care of ambiguous unicode characters
        # First we fix any character issues due to encoding
        self._fix_text_encoding(columns)

        # Secondly, we replace all characters that were not corrected in
        # the first step. They cannot be deleted since the words will change
        self._find_and_replace_non_ascii(columns)

        # Thirdly, some characters should not be replaced since they do not
        # contribute to overall context, thus can be deleted
        self.remove_specified_characters_from_dataframe()

        # In final step where we check if our data is clean, otherwise,
        # it will print characters that are causing problems
        self._show_ambiguous_unicode(columns)
        # ----------------------------------------------

        # convert to lowercase text
        self._to_lowercase(columns)

        # Check for extra whitespaces
        self._delete_extra_whitespace(columns)

    def _encode_labels(self):
        """
        Convert bias_rating (left, right, neutral) to numerical labels.
        """
        bias_mapping = {'left': 0, 'center': 1, 'right': 2}
        self._dataframe['bias_rating'] = self._dataframe['bias_rating'].map(bias_mapping)
        pass

    def preprocess(self):
        """
        Apply all preprocessing steps to a single data point (title, heading, source, tags, bias_rating).
        """
        print("Welcome to data preprocessing pipeline!")

        # select only heading, text and label
        self._extract_columns()

        # check for missing values
        self._check_missing_values()

        # check for duplicates
        self._check_duplicates()

        # combine "heading" and "text" columns
        self._combine_columns()

        # encode bias_rating into a numerical label
        self._encode_labels()

        # normalize only heading and text (ascii characters, lowercase, extra whitespaces)
        self._normalize_text()  

    def __getitem__(self, index: int):
        """
        Returns a row from the DataFrame at the given index.
        """
        if index >= len(self._dataframe) or index < 0:
            raise IndexError("Index out of range.")
        return self._dataframe.iloc[index]
    
    def __len__(self) -> int:
        return len(self._dataframe)

    def get_df(self):
        return self._dataframe


if __name__ == "__main__":
    pass