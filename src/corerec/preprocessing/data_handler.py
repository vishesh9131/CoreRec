# preprocessing/data_handler.py
from corerec.format_master import FormatMaster
from corerec.preprocessing.config_loader import ConfigLoader
import pandas as pd
import logging
import chardet

class DataHandler:
    """Handles data loading, format detection, preprocessing, and validation using FormatMaster."""
    
    def __init__(self, config_path: str = None):
        self.format_master = FormatMaster(config_path=config_path)
        logging.info("Initialized DataHandler with FormatMaster.")
    
    def detect_encoding(self, file_path):
        """Detects the encoding of a file."""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            return result['encoding']
    
    def load_and_process(self, file_path, ignore_errors=False):
        """Loads and preprocesses the data file."""
        try:
            # Detect file encoding
            encoding = self.detect_encoding(file_path)
            logging.info(f"Detected encoding: {encoding}")

            # Read the file with detected encoding
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep='::',
                engine='python',
                names=['item_id', 'item_description', 'genre'],
                on_bad_lines='skip' if ignore_errors else 'error'
            )
            # Perform any additional preprocessing steps
            return df
        except Exception as e:
            logging.error(f"An error occurred during the pipeline execution: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error