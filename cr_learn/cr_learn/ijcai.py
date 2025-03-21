import pandas as pd
import json
from typing import Dict, List, Any
import os
from cr_learn.utils.gdrive_downloader import check_and_download
from cr_learn.utils.cr_cache_path import path

'''
THE DATAET IS AUTHORISED AND BEING PICKED FROM RESPECTED CITATION [Responsible : @vishesh9131]
        title={# IJCAI-16 Brick-and-Mortar Store Recommendation Dataset}
        url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=53}
        author={Tianchi},
        year={2018}
'''
# path = 'CRLearn/CRDS/Ijcai_16'
default_file_path = 'CRLearn/CRDS/Ijcai_16'

# train_cluster = os.path.join(path,'train_format1.csv')
# test_cluster = os.path.join(path,'test_format1.csv')

# def load_users(cluster: str) -> pd.DataFrame:
#     return pd.read_csv(cluster,sep=',',engine='python',names=['user_id','age_range','gender'],encoding='latin-1')

# users_df=load_users(train_cluster)

def load(data_path: str = path, limit_rows=100) -> Dict[str, Any]:
    """Load and process users, ratings, and movies data from the specified path."""
    # Create directory if it doesn't exist
    check_and_download('ijcai', base_path=data_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    data_path = os.path.abspath(data_path)

    # biscuit : reading usr dataset
    column_names_users = ['user_id', 'age_range', 'gender']
    users = pd.read_csv(
        os.path.join(data_path, 'user_info_format1.csv'),
        sep=',',
        engine='python',
        names=column_names_users,
        encoding='latin-1'
    )

    # Use consistent column names for merchant data
    column_names_merchant = ['user_id', 'merchant_id', 'label']
    
    # biscuit : reading merchant train dataset
    merchant_train_data = pd.read_csv(
        os.path.join(data_path, 'train_format1.csv'),
        sep=',',
        engine='python',
        names=column_names_merchant,
        encoding='latin-1'
    )
    # Ensure label is numeric
    merchant_train_data['label'] = pd.to_numeric(merchant_train_data['label'], errors='coerce')

    # biscuit : reading merchant test dataset
    merchant_test_data = pd.read_csv(
        os.path.join(data_path, 'test_format1.csv'),
        sep=',',
        engine='python',
        names=column_names_merchant,
        encoding='latin-1'
    )
    # Ensure label is numeric
    merchant_test_data['label'] = pd.to_numeric(merchant_test_data['label'], errors='coerce')

    # biscuit : reading user log dataset [its quite massive]
    user_log = pd.read_csv(
        os.path.join(data_path, 'test_format1.csv'),
        sep=',',
        engine='python',
        names=column_names_merchant,
        encoding='latin-1',
        nrows=limit_rows
    )
 
    user_merchant_interaction = merchant_test_data.groupby('user_id')['merchant_id'].apply(list).to_dict()
    
    # Build Merchant Features - using a simpler approach to avoid aggregation errors
    merchant_features = {}
    
    # Combine train and test data for better feature creation
    all_merchant_data = pd.concat([merchant_train_data, merchant_test_data])
    
    # Make sure all columns are properly typed
    all_merchant_data['merchant_id'] = pd.to_numeric(all_merchant_data['merchant_id'], errors='coerce')
    all_merchant_data['label'] = pd.to_numeric(all_merchant_data['label'], errors='coerce')
    
    # Use a simpler approach with separate aggregations
    merchant_ids = all_merchant_data['merchant_id'].unique()
    
    for merchant_id in merchant_ids:
        merchant_data = all_merchant_data[all_merchant_data['merchant_id'] == merchant_id]
        
        # Calculate features manually
        total_interactions = len(merchant_data)
        unique_users = merchant_data['user_id'].nunique()
        
        # Only calculate conversion rate if there are valid labels
        if merchant_data['label'].notna().any():
            conversion_rate = merchant_data['label'].mean()
        else:
            conversion_rate = 0.0
        
        merchant_features[merchant_id] = {
            'conversion_rate': float(conversion_rate),
            'interaction_level': 'high' if total_interactions > 10 else 'low',  # Using a fixed threshold
            'user_diversity': 'high' if unique_users > 5 else 'low'  # Using a fixed threshold
        }

    return {
        'users': users,
        'merchant_train': merchant_train_data,
        'merchant_test': merchant_test_data,
        'user_log': user_log,
        'user_merchant_interaction': user_merchant_interaction,
        'merchant_features': merchant_features
    }



# def prepare_embedding_data(data_path: str=path) -> List[List[str]]:
#     """
#     Prepare data for embedding training by creating sequences of merchant interactions
#     and their features.

#     Parameters:
#     - data_path (str): Path to the IJCAI dataset files

#     Returns:
#     - List[List[str]]: List of sequences for embedding training
#     """
#     # Load the necessary data
#     data = load(data_path)
#     merchant_train = data['merchant_train']
#     merchant_features = data['merchant_features']
    
#     sequences = []
    
#     # Create sequences based on user-merchant interactions
#     user_sequences = merchant_train.groupby('user_id')['merchant_id'].apply(list)
#     sequences.extend([list(map(str, merchant_seq)) for merchant_seq in user_sequences])
    
#     # Create sequences based on merchant features
#     for merchant_id, features in merchant_features.items():
#         feature_seq = [
#             f"merchant_{merchant_id}",
#             f"conv_{features['conversion_rate']:.2f}",
#             f"interact_{features['interaction_level']}",
#             f"diversity_{features['user_diversity']}"
#         ]
#         sequences.append(feature_seq)
    
#     return sequences
