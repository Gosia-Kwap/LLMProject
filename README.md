# News Bias Classification: Encoder vs Decoder Models

This project focuses on analyzing and fine-tuning large language models (LLMs) to classify political bias in text. The project includes various experiments with models such as RoBERTa and Gemma, along with prompt engineering techniques to improve classification performance.


- **data/**: Folder with the datasets.
  - **raw/**: Folder with the raw data.
  - **processed/**: Folder with the processed data.

- **src/**: Folder with the code.
  - **data/**: Folder with the data pipeline scripts.
    - `csv_finder.py`: Utility script to locate specific CSV files.
    - `data_loader.py`: Script for loading and preparing data.
    - `data_preprocessing.py`: Script for data cleaning and preprocessing.
  - **models/**: Folder with model-related scripts (if applicable).
  - **utils/**: Folder with utility functions (if applicable).
  - `main.py`: Main script to train and evaluate models.

- **notebooks/**: Folder with the Jupyter notebooks.
  - **Data analysis/**: Notebook for model comparison.
    - `model_comparison.ipynb`
  - **Gemma/**: Notebooks for Gemma model experiments.
    - `gemma_baseline.ipynb`: Baseline model notebook.
    - `gemma_lora.ipynb`: LoRA fine-tuning for Gemma.
    - `gemma-fine-tuned.ipynb`: Fully fine-tuned Gemma model.
    - `prompt_eng_no_center.ipynb`: Prompt engineering notebook.
    - `prompt_eng_version_2.ipynb`: Second version of prompt engineering.
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

- **tests/**: Folder with test scripts (if applicable).

- **deployment/**: Folder with deployment scripts (if applicable).

- **logs/**: Folder with log files (if applicable).

- **config/**: Folder with configuration files (if applicable).

- **README.md**: File with project instructions.

- **requirements.txt**: File with the list of dependencies.

