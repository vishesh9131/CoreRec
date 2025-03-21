import pandas as pd 
from typing import Dict, List, Any
import os
from cr_learn.utils.gdrive_downloader import check_and_download
from cr_learn.utils.cr_cache_path import path

# path = 'CRLearn/CRDS/Tmall'
default_file_path = 'CRLearn/CRDS/Tmall'



test_cluster = os.path.join(path, 'ijcai2016_koubei_test')
train_cluster = os.path.join(path, 'ijcai2016_koubei_train')
merchant_info = os.path.join(path, 'ijcai2016_merchant_info')
taobao_file = os.path.join(path, 'ijcai2016_taobao.csv')


def load_test_cluster(test_cluster: str) -> pd.DataFrame:
    return pd.read_csv(test_cluster, sep=',', names=['use_ID', 'loc_ID'], encoding='utf-8')

def load_train_cluster(train_cluster: str) -> pd.DataFrame:
    return pd.read_csv(train_cluster, sep=',', names=['use_ID', 'mer_ID', 'loc_ID', 'time'], encoding='utf-8')

def load_merchant_info(merchant_info: str) -> pd.DataFrame:
    return pd.read_csv(merchant_info, sep=',', names=['mer_ID', 'budget', 'loc_list'], encoding='utf-8')

def load_taobao(taobao_file: str, nrows: int = 200) -> pd.DataFrame:
    return pd.read_csv(taobao_file, sep=',', nrows=nrows, encoding='utf-8')

def load(data_path: str = path) -> Dict[str, pd.DataFrame]:
    """Load and process test, train, merchant info, and taobao data from the specified path."""
    # Check and download missing files
    check_and_download('tmall', base_path=data_path)

    # Ensure the data_path is relative to the current working directory
    data_path = os.path.abspath(data_path)  # Convert to absolute path

    test_df = load_test_cluster(os.path.join(data_path, 'ijcai2016_koubei_test'))
    train_df = load_train_cluster(os.path.join(data_path, 'ijcai2016_koubei_train'))
    merchant_df = load_merchant_info(os.path.join(data_path, 'ijcai2016_merchant_info'))
    taobao_df = load_taobao(os.path.join(data_path, 'ijcai2016_taobao.csv'))

    return {
        'test': test_df,
        'train': train_df,
        'merchant_info': merchant_df,
        'taobao': taobao_df
    }

def info(data: Dict[str, pd.DataFrame]):
    """Print detailed information about each dataset."""
    for name, df in data.items():
        print(f"\n{name.capitalize()} DataFrame:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("First few rows:")
        print(df.head())
        print("\nColumn data types:")
        print(df.dtypes)

# def main():
#     """Main function to load and process the Tmall dataset."""
#     print("Loading Tmall dataset...")
#     data = load()
    
#     # Display basic information about each dataset
#     for name, df in data.items():
#         print(f"\n{name.capitalize()} DataFrame:")
#         print(df.head())
#         print(f"Number of rows: {len(df)}")
#         print(f"Number of columns: {len(df.columns)}")

# if __name__ == "__main__":
#     main()

