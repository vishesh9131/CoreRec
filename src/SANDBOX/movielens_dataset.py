import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, num_users, num_items):
        self.data = pd.read_csv(ratings_file, sep='::', header=None, engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Encode user_id and item_id
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        self.data['user_id'] = user_encoder.fit_transform(self.data['user_id'])
        self.data['item_id'] = item_encoder.fit_transform(self.data['item_id'])
        
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = self.data.iloc[idx, 0]
        item_id = self.data.iloc[idx, 1]
        rating = self.data.iloc[idx, 2]
        
        # Create one-hot encoded user and item vectors
        user_vector = np.zeros(self.num_users)
        item_vector = np.zeros(self.num_items)
        user_vector[user_id] = 1
        item_vector[item_id] = 1
        
        return torch.tensor(user_vector, dtype=torch.float32), torch.tensor(item_vector, dtype=torch.float32), torch.tensor(rating, dtype=torch.float32)
