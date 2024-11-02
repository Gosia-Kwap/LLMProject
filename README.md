# News Bias Classification: Encoder vs Decoder Models

This project focuses on analyzing and fine-tuning large language models (LLMs) to classify political bias in text. The project includes various experiments with models such as RoBERTa and Gemma, along with prompt engineering techniques to improve classification performance.

Here is structure of our project:

- **data/**: Folder with the datasets.
  - **raw/**: Folder with the raw data.
  - **processed/**: Folder with the processed data.

- **src/**: Folder with the code.
  - **data/**: Folder with the data pipeline scripts.
    - `csv_finder.py`: Utility script to locate specific CSV files.
    - `data_loader.py`: Script for loading and preparing data.
    - `data_preprocessing.py`: Script for data cleaning and preprocessing.
    - 'main.py': script to preprocess data

- **notebooks/**: Folder with the Jupyter notebooks.
  - **Data analysis/**: Notebook for model comparison.
    - `model_comparison.ipynb`
  - **Gemma/**: Notebooks for Gemma model experiments.
    - `gemma_baseline.ipynb`: Baseline model notebook.
    - `gemma_lora.ipynb`: LoRA fine-tuning for Gemma.
    - `gemma-fine-tuned.ipynb`: Fully fine-tuned Gemma model.
    - `prompt_eng_no_center.ipynb`: Prompt engineering no center.
    - `prompt_eng_version_2.ipynb`: Includes 2 versions of prompt engineering.
  - **Roberta/**: Notebooks for RoBERTa model experiments.
    - `Roberta_lora.ipynb`: LoRA fine-tuning for RoBERTa.
    - `Roberta-baseline.ipynb`: Baseline model notebook.
    - `Roberta-fine-tuned.ipynb`: Fully fine-tuned RoBERTa model.

- **results/**: Folder with model performance results.
  - `gemma_fine_tune.csv`: Results for fine-tuning Gemma.
  - `gemma_lora.csv`: Results for LoRA fine-tuning on Gemma.
  - `political_bias_results_prompt_1.json`: JSON results for prompt engineering (first version).
  - `political_bias_results_prompt_2.json`: JSON results for prompt engineering (second version).
  - `prompt_no_center.json`: JSON results for prompt engineering without center bias.
  - `roberta_fine_tuning.csv`: Results for fine-tuning RoBERTa.
  - `roberta_lora.csv`: Results for LoRA fine-tuning on RoBERTa.

- **README.md**

- **requirements.txt**: File with the list of dependencies.



## Getting Started

Prerequisites:

Python 3.8 or higher

Required packages listed in requirements.txt

Install dependencies with: pip install -r requirements.txt

## Project Overview

This project uses both fine-tuning and prompt engineering methods for political bias classification in news articles. The experiments involve:

* Baseline and Fine-Tuning of LLMs such as RoBERTa and Gemma.
* Prompt Engineering to guide model outputs for bias detection.
* LoRA (Low-Rank Adaptation) as a PEFT technique to improve model efficiency during fine-tuning.

# Limitations

 The systems tailored to classifying the political bias might suffer greatly from the lack of reliability or scalability. In particular, our system limitation is the spatially limited source of the data – the political environment of the USA. In the context of USA's politics, the distinction "left" or "right" might take the form of e.g. "pro-democrat" or "pro-republican", which loses scalability for other political contexts, outside of this environment. 