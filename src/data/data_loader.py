import os
import pandas as pd
import csv
from data_preprocessing import TextPreprocessor
from typing import Union


class DataSetLoader():
    '''
    Callable class, loads the data into a csv file at a specified location.
    '''
    def __call__(self, dataset: Union['TextPreprocessor', pd.DataFrame], path:str):
        file_path = os.path.join(os.getcwd(), path)
        if isinstance(dataset, TextPreprocessor):
            dataset._dataframe.to_csv(file_path, index = False)
        elif isinstance(dataset, pd.DataFrame):
            dataset.to_csv(file_path, index = False)
        else:
            raise ValueError("Can only load objects of type pd.TextPreprocessor and TextPreprocessor.")
