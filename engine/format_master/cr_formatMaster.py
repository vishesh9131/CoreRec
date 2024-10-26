"""
This format detection pipeline will be later plugged into corerec's utilities.

This script provides a complete pipeline for data processing, including loading, 
format detection, preprocessing, and validation of data.

Functions:
- load_data(file_path): Loads data from the specified file path into a DataFrame.
- detect_format(df): Detects the format of the given DataFrame.
- preprocess_data(df, data_format): Preprocesses the DataFrame based on the detected format.
- validate_data(preprocessed_df): Validates the preprocessed DataFrame to ensure it meets required standards.
- format_detection_pipeline(file_path): Orchestrates the entire process of loading, detecting, 
  preprocessing, and validating data from a given file path.

config dictionary:
- parallel_processing: bool, whether to use parallel processing.
- log_level: str, the level of logging to use.
- chunk_size: int, the size of chunks to use for parallel processing.
- missing_value_strategy: str, the strategy to use for handling missing values.
- scaling_method: str, the method to use for scaling the data.
- validation_rules: dict, the rules to use for validating the data.
- report_format: str, the format to use for the report.
- log_file: str, the file to use for logging.
- monitoring: bool, whether to use monitoring.
- num_workers: int, the number of workers to use for parallel processing.
- distributed_backend: str, the distributed backend to use.
- custom_steps: list, the custom steps to use.

Example config:
config = {
    'parallel_processing': True,
    'log_level': 'INFO',
    'chunk_size': 10000,
    'missing_value_strategy': 'fill_mean',
    'scaling_method': 'standard',
    'validation_rules': {'max_null_percentage': 0.1},
    'report_format': 'json',
    'log_file': 'pipeline.log',
    'monitoring': True,
    'num_workers': 4,
    'distributed_backend': 'dask',
    'custom_steps': ['step1', 'step2']
    }

Author: Vishesh Yadav
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
import logging
import dask.dataframe as dd  # Dask for handling large datasets
from dask.distributed import Client
import yaml
import glob
from fuzzywuzzy import fuzz, process

from engine.format_master.format_library import *
from ds_format_loader import load_data, detect_format, preprocess_data, validate_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Load configuration from a JSON or YAML file."""
    with open(config_path, 'r') as file:
        if config_path.endswith('.json'):
            return json.load(file)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(file)
        else:
            raise ValueError("Unsupported configuration file format.")

def setup_dask_client(config):
    """Setup Dask client for distributed processing."""
    if config.get('distributed_backend') == 'dask':
        client = Client(n_workers=config.get('num_workers', 4))
        logging.info("Dask client set up with {} workers.".format(config.get('num_workers', 4)))
        return client
    return None

def find_file(file_name, search_paths, extensions=None, threshold=80):
    """Search for a file in multiple directories with different possible names and extensions."""
    if extensions is None:
        extensions = ['.csv', '.txt', '.json']  # Add more extensions as needed

    possible_names = [file_name, file_name.lower(), file_name.upper(), file_name.replace(' ', '_')]
    all_files = []

    # Collect all files in the search paths
    for path in search_paths:
        for root, dirs, files in os.walk(path):
            for file in files:
                all_files.append(os.path.join(root, file))

    # Try to find the file using exact and fuzzy matching
    for name in possible_names:
        for ext in extensions:
            target_name = f"{name}{ext}"
            # Exact match
            for file_path in all_files:
                if os.path.basename(file_path) == target_name:
                    logging.info(f"Exact match found: {file_path}")
                    return file_path

            # Fuzzy match
            matches = process.extractBests(target_name, all_files, scorer=fuzz.ratio, score_cutoff=threshold)
            if matches:
                best_match = matches[0][0]
                logging.info(f"Fuzzy match found: {best_match}")
                return best_match

    logging.warning(f"File {file_name} not found in specified directories.")
    return None

def detect(data, config=None, custom_preprocess=None):
    """
    Advanced pipeline to load, detect, preprocess, and validate data.
    
    Parameters:
    - data: DataFrame or str, the data or path to the data file.
    - config: dict, optional configuration for the pipeline.
    - custom_preprocess: function, optional custom preprocessing function.
    
    Returns:
    - preprocessed_df: DataFrame, the preprocessed data.
    """
    try:
        if isinstance(data, str):
            logging.info(f"Searching for data file: {data}")
            search_paths = ['.', './data', './datasets']  # Add more directories as needed
            data_path = find_file(data, search_paths)
            
            if data_path:
                logging.info(f"Loading data from {data_path}")
                if data_path.endswith('.csv'):
                    # Specify dtypes to avoid dtype inference issues
                    dtype_spec = {'isbn': 'object'}  # Specify other columns as needed
                    df = dd.read_csv(data_path, blocksize=config.get('chunk_size', '64MB'), dtype=dtype_spec)
                    is_dask = True
                else:
                    df = load_data(data_path)
                    is_dask = False
                
                if is_dask:
                    df = df.compute()
            else:
                logging.error("Data file not found. Exiting the pipeline.")
                return pd.DataFrame()
        else:
            df = data
            logging.info("Data provided directly as DataFrame")

        if df.empty:
            warnings.warn("The loaded DataFrame is empty.")
            return df

        logging.info(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")

        data_format = detect_format(df)
        if data_format is None:
            raise ValueError("Data format detection failed")
        logging.info(f"Detected data format: {data_format}")

        if custom_preprocess:
            logging.info("Applying custom preprocessing function")
            preprocessed_df = custom_preprocess(df, data_format)
        else:
            logging.info("Applying standard preprocessing")
            preprocessed_df = preprocess_data(df, data_format)

        if preprocessed_df is None:
            raise ValueError("Preprocessing failed")

        logging.info("Validating preprocessed data")
        validate_data(preprocessed_df)

        logging.info(f"Preprocessed data has {preprocessed_df.shape[0]} rows and {preprocessed_df.shape[1]} columns")

        return preprocessed_df

    except Exception as e:
        logging.error(f"An error occurred during the pipeline execution: {e}")
        return pd.DataFrame()

# if __name__ == "__main__":
#     config_path = 'config.yaml'  # Path to your configuration file
#     config = load_config(config_path)
    
#     dask_client = setup_dask_client(config)
    
#     preprocessed_data = detect('data/books.csv', config=config)
    
#     if dask_client:
#         dask_client.close()
