import os
import pandas as pd


def find_dataset_dir(path):
    """
    Automatically constructs the dataset directory path based on the current working directory and variable.
    """
    cwd = os.getcwd()  # current working directory
    dataset_dir = os.path.join(cwd, path)

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"The directory {dataset_dir} does not exist.")

    if not os.listdir(dataset_dir):
        raise ValueError(f"The directory {dataset_dir} is empty.")
    
    return dataset_dir

def load_data(dataset_dir):
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

if __name__ == "__main__":
    pass
