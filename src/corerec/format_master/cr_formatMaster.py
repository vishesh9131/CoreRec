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

from corerec.format_master.ds_format_loader import load_data, detect_format, preprocess_data, validate_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FormatMaster:
    """Encapsulates the data loading, format detection, preprocessing, and validation pipeline."""
    
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path:
            self.config = self.load_config(config_path)
            self.dask_client = self.setup_dask_client(self.config)
        else:
            self.dask_client = None
        logging.info("Initialized FormatMaster.")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from a JSON or YAML file."""
        with open(config_path, 'r') as file:
            if config_path.endswith('.json'):
                config = json.load(file)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(file)
            else:
                raise ValueError("Unsupported configuration file format.")
        logging.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_dask_client(self, config: dict):
        """Setup Dask client for distributed processing."""
        if config.get('distributed_backend') == 'dask':
            client = Client(n_workers=config.get('num_workers', 4))
            logging.info(f"Dask client set up with {config.get('num_workers', 4)} workers.")
            return client
        return None
    
    def find_file(self, filename: str, search_paths: list) -> str:
        """Search for a file in the specified directories."""
        for path in search_paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                logging.info(f"Found file: {full_path}")
                return full_path
        logging.warning(f"File {filename} not found in specified search paths.")
        return None
    
    def detect(self, data, config: dict = None, custom_preprocess: callable = None) -> pd.DataFrame:
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
                data_path = self.find_file(data, search_paths)
                
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
