# mimicing crds_health

import pandas as pd
import logging
from typing import Dict, Any
import numpy as np

from cr_learn import beibei, ijcai, library_thing, ml_1m, rees46, steam_games

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_dataframe_info(df: pd.DataFrame, name: str):
    """Print detailed information about a DataFrame"""
    logging.info(f"\n{'-'*20} {name} DataFrame Analysis {'-'*20}")
    logging.info(f"Shape: {df.shape}")
    logging.info("\nColumns:")
    for col in df.columns:
        non_null = df[col].count()
        dtype = df[col].dtype
        unique_vals = df[col].nunique()
        logging.info(f"- {col}: {dtype} | Non-null: {non_null} | Unique values: {unique_vals}")
    
    logging.info("\nSample Data (first 3 rows):")
    logging.info(df.head(3))
    
    logging.info("\nBasic Statistics:")
    for col in df.select_dtypes(include=[np.number]).columns:
        stats = df[col].describe()
        logging.info(f"\n{col}:")
        logging.info(stats)

def test_beibei():
    """Test Beibei dataset loading and functions"""
    logging.info("\n=== Testing Beibei Dataset ===")
    try:
        datasets = beibei.load()
        for name, dataset in datasets.items():
            logging.info(f"\n{name} Dataset:")
            logging.info(f"Shape: {dataset.data.shape}")
            logging.info(f"Number of users: {dataset.get_user_count()}")
            logging.info(f"Number of items: {dataset.get_item_count()}")
            logging.info(f"Total interactions: {dataset.get_interaction_count()}")
            logging.info(f"Sparsity: {dataset.get_sparsity():.4f}")
            
            # Show popular items
            logging.info("\nTop 5 Popular Items:")
            logging.info(dataset.get_popular_items(5))
            
            # Show active users
            logging.info("\nTop 5 Active Users:")
            logging.info(dataset.get_active_users(5))
            
    except Exception as e:
        logging.error(f"Error in Beibei dataset: {str(e)}")

def test_ijcai():
    """Test IJCAI dataset loading and functions"""
    logging.info("\n=== Testing IJCAI Dataset ===")
    try:
        data = ijcai.load()
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print_dataframe_info(value, key)
            elif isinstance(value, dict):
                logging.info(f"\n{key} Dictionary:")
                logging.info(f"Number of entries: {len(value)}")
                logging.info("Sample entries (first 3):")
                sample_items = list(value.items())[:3]
                for k, v in sample_items:
                    logging.info(f"{k}: {v}")
    except Exception as e:
        logging.error(f"Error in IJCAI dataset: {str(e)}")

def test_library_thing():
    """Test LibraryThing dataset loading and functions"""
    logging.info("\n=== Testing LibraryThing Dataset ===")
    try:
        # Test mapping loading
        mapping_df = library_thing.load_mapping()
        print_dataframe_info(mapping_df, "Mapping")
        
        # Test main load function
        data = library_thing.load()
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print_dataframe_info(value, key)
            elif isinstance(value, dict):
                logging.info(f"\n{key} Dictionary:")
                logging.info(f"Number of entries: {len(value)}")
                logging.info("Sample entries (first 3):")
                sample_items = list(value.items())[:3]
                for k, v in sample_items:
                    logging.info(f"{k}: {v}")
    except Exception as e:
        logging.error(f"Error in LibraryThing dataset: {str(e)}")

def test_ml_1m():
    """Test MovieLens 1M dataset loading and functions"""
    logging.info("\n=== Testing MovieLens 1M Dataset ===")
    try:
        data = ml_1m.load()
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print_dataframe_info(value, key)
            elif isinstance(value, dict):
                logging.info(f"\n{key} Dictionary:")
                logging.info(f"Number of entries: {len(value)}")
                logging.info("Sample entries (first 3):")
                sample_items = list(value.items())[:3]
                for k, v in sample_items:
                    logging.info(f"{k}: {v}")
    except Exception as e:
        logging.error(f"Error in MovieLens 1M dataset: {str(e)}")

def test_rees46():
    """Test REES46 dataset loading and functions"""
    logging.info("\n=== Testing REES46 Dataset ===")
    try:
        data = rees46.load(sample_size=0.1)
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print_dataframe_info(value, key)
    except Exception as e:
        logging.error(f"Error in REES46 dataset: {str(e)}")

def test_steam_games():
    """Test Steam Games dataset loading and functions"""
    logging.info("\n=== Testing Steam Games Dataset ===")
    try:
        df = steam_games.load_df()
        print_dataframe_info(df, "Steam Games")
        
        # Additional Steam-specific analysis
        logging.info("\nGenre Analysis:")
        genre_counts = df['genres'].apply(len).describe()
        logging.info(f"Genres per game statistics:\n{genre_counts}")
        
        logging.info("\nPrice Analysis:")
        price_stats = df['price'].describe()
        logging.info(f"Price statistics:\n{price_stats}")
        
        logging.info("\nTop 10 Publishers by Game Count:")
        publisher_counts = df['publisher'].value_counts().head(10)
        logging.info(publisher_counts)
        
    except Exception as e:
        logging.error(f"Error in Steam Games dataset: {str(e)}")

def run_all_tests():
    """Run all dataset tests"""
    logging.info("Starting detailed dataset tests...")
    
    tests = [
        test_beibei,
        test_ijcai,
        test_library_thing,
        test_ml_1m,
        test_rees46,
        test_steam_games
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            logging.error(f"Error in {test.__name__}: {str(e)}")
    
    logging.info("Dataset tests completed.")

if __name__ == "__main__":
    run_all_tests() 