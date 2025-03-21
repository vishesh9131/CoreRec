import pandas as pd
import json
from typing import Dict, List, Any
import os
import ast
from cr_learn.utils.gdrive_downloader import check_and_download, get_dataset_path
from cr_learn.utils.cr_cache_path import path

# Define the base path
# path = 'CRLearn/CRDS/steam_games'
default_file_path = 'CRLearn/CRDS/steam_games'

def load_mapping() -> pd.DataFrame:
    """Load the mapping data for Steam games."""
    # Read the mapping file if it exists, otherwise return empty DataFrame
    mapping_path = os.path.join(get_dataset_path('steam_games', base_path=path), 'mapping.csv')
    if os.path.exists(mapping_path):
        return pd.read_csv(mapping_path)
    return pd.DataFrame()

def load1(data_path: str = path) -> Dict[str, Any]:
    """Load Steam games data from JSON file.
    
    Args:
        data_path: Path to the dataset directory
        
    Returns:
        Dict containing:
            - games: DataFrame with game information
            - genres: List of unique genres
            - tags: List of unique tags
    """
    # Ensure the directory exists and download files if needed
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    
    # Download the dataset if needed
    check_and_download('steam_games', base_path=data_path)
    
    # Get the actual dataset path where files were downloaded
    dataset_path = get_dataset_path('steam_games', base_path=data_path)
    
    # Load the games data
    games_data = []
    json_path = os.path.join(dataset_path, 'steam_games.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Use ast.literal_eval to safely evaluate Python dictionary string
                game = ast.literal_eval(line.strip())
                games_data.append(game)
            except (SyntaxError, ValueError) as e:
                if line.strip():  # Only print error for non-empty lines
                    print(f"Error parsing line: {e}")
                continue
    
    # Convert to DataFrame
    games_df = pd.DataFrame(games_data)
    
    # Extract unique genres and tags
    all_genres = set()
    all_tags = set()
    
    for _, row in games_df.iterrows():
        if isinstance(row.get('genres'), list):
            all_genres.update(row['genres'])
        if isinstance(row.get('tags'), list):
            all_tags.update(row['tags'])
    
    return {
        'games': games_df,
        'genres': sorted(list(all_genres)),
        'tags': sorted(list(all_tags))
    }

def load(data_path: str = path) -> pd.DataFrame:
    """Load Steam games data and return as DataFrame.
    
    Args:
        data_path: Path to the dataset directory
    """
    # Ensure the directory exists and download files if needed
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    
    # Download the dataset if needed
    check_and_download('steam_games', base_path=data_path)
    
    # Get the actual dataset path where files were downloaded
    dataset_path = get_dataset_path('steam_games', base_path=data_path)
    
    games_data = []
    json_path = os.path.join(dataset_path, 'steam_games.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                game = ast.literal_eval(line.strip())
                games_data.append(game)
            except (SyntaxError, ValueError) as e:
                continue
    
    return pd.DataFrame(games_data)

if __name__ == "__main__":
    # Set data path
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CRDS', 'steam_games')
    
    # Load and display DataFrame info
    df = load_df(data_path)
    
    print("\nSteam Games DataFrame Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    print("\nSample Data (first 3 rows):")
    print(df.head(3))
    
    print("\nDataFrame Info:")
    print(df.info())
    
    print("\nUnique Values in Key Columns:")
    print(f"Number of unique publishers: {df['publisher'].nunique()}")
    print(f"Number of unique developers: {df['developer'].nunique()}")
    print(f"Number of games with metascores: {df['metascore'].count()}") 
    