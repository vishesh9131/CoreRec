import pandas as pd
import os
from typing import Dict, Any
import json
import logging
from cr_learn.utils.cr_cache_path import path
from cr_learn.utils.gdrive_downloader import get_dataset_path, check_and_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the paths
# Use the dataset path from gdrive_downloader
def get_rees46_path(data_path=None):
    """Get the path to the REES46 dataset."""
    # Ensure the dataset is downloaded
    check_and_download('rees46', base_path=data_path)
    # Get the path to the dataset
    return get_dataset_path('rees46', base_path=data_path)

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    if not os.path.exists(file_path):
        logging.error(f"The file {file_path} does not exist.")
        return False
    return True

def load_csv(file_path: str, use_columns: list = None, nrows: int = None) -> pd.DataFrame:
    """Generic function to load a CSV file with safety checks."""
    if not check_file_exists(file_path):
        return pd.DataFrame()  # Return empty DataFrame instead of raising error
    
    try:
        return pd.read_csv(
            file_path, 
            usecols=use_columns, 
            nrows=nrows,  # Add row limit
            low_memory=False,
            on_bad_lines='skip'  # Skip problematic lines
        )
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def load_events(data_path=None, sample_size: float = 1.0, use_columns: list = None, nrows: int = None) -> pd.DataFrame:
    """Load and optionally sample events data."""
    try:
        rees46_path = get_rees46_path(data_path)
        events_file = os.path.join(rees46_path, 'events.csv')
        df = load_csv(events_file, use_columns, nrows)
        if len(df) > 0 and sample_size < 1.0:
            df = df.sample(frac=sample_size)
        return df
    except Exception as e:
        logging.error(f"Error loading events: {str(e)}")
        return pd.DataFrame()

def preprocess_events(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the events DataFrame."""
    if len(df) == 0:
        return df
        
    try:
        df = df.copy()
        df.fillna({'brand': 'unknown'}, inplace=True)
        if 'event_time' in df.columns:
            df['event_time'] = pd.to_datetime(df['event_time'])
        if 'price' in df.columns:
            df = df[df['price'] >= 10]
        return df
    except Exception as e:
        logging.error(f"Error preprocessing events: {str(e)}")
        return df

def load_direct_msg_data(data_path=None, limit_rows: int = 100) -> Dict[str, pd.DataFrame]:
    """Load all datasets from the direct_msg folder with a row limit."""
    datasets = {}
    files = ['campaigns.csv', 'client_first_purchase_date.csv', 'holidays.csv', 'messages-demo.csv']
    
    try:
        rees46_path = get_rees46_path(data_path)
        dm_path = os.path.join(rees46_path, 'direct_msg')
        
        # Create the directory if it doesn't exist
        os.makedirs(dm_path, exist_ok=True)
        
        for file in files:
            file_path = os.path.join(dm_path, file)
            if check_file_exists(file_path):
                df = load_csv(file_path, nrows=limit_rows)
                if len(df) > 0:
                    datasets[file.split('.')[0]] = df
    except Exception as e:
        logging.error(f"Error loading direct message data: {str(e)}")
    
    return datasets

def load(data_path: str = None, sample_size: float = 0.1, use_columns: list = None, nrows: int = 1000) -> Dict[str, pd.DataFrame]:
    """Load and preprocess all data with safety limits."""
    try:
        # Ensure the dataset is downloaded
        check_and_download('rees46', base_path=data_path)
        
        # Load events data
        events = load_events(data_path, sample_size, use_columns, nrows)
        
        # Process events data
        processed_events = preprocess_events(events)
        
        # Load direct message data
        direct_msg_data = load_direct_msg_data(data_path, limit_rows=nrows)
        
        # Combine results
        result = {'events': processed_events}
        result.update(direct_msg_data)
        
        return result
        
    except Exception as e:
        logging.error(f"Error in load function: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {'events': pd.DataFrame()}

def load_config(data_path: str = None) -> Dict[str, Any]:
    """Load configuration from the specified path."""
    try:
        rees46_path = get_rees46_path(data_path)
        config_path = os.path.join(rees46_path, 'context_config.json')
        
        # Create an empty config file if it doesn't exist
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found at {config_path}. Creating empty config.")
            with open(config_path, 'w') as f:
                json.dump({}, f)
        
        with open(config_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file.")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading configuration: {e}")
        return {}

if __name__ == "__main__":
    # Test with small sample size and row limit
    logging.info("Testing REES46 data loading...")
    
    try:
        # Test events loading
        events_sample = load_events(sample_size=0.01, nrows=1000)
        logging.info(f"Loaded events sample shape: {events_sample.shape}")
        
        # Test direct message data loading
        msg_data = load_direct_msg_data(limit_rows=100)
        for name, df in msg_data.items():
            logging.info(f"Loaded {name} shape: {df.shape}")
        
        # Test full load
        all_data = load(sample_size=0.01, nrows=1000)
        for name, df in all_data.items():
            logging.info(f"Final {name} shape: {df.shape}")
            
    except KeyboardInterrupt:
        logging.info("Loading interrupted by user")
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
