import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class ActorDataset(Dataset):
    def __init__(self, adjacency_matrix, features):
        self.adj_matrix = adjacency_matrix
        self.features = features

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.adj_matrix[idx]

def load_and_preprocess(filepath):
    # Load the dataset
    actors_df = pd.read_csv(filepath)
    
    # Fill NaN values with 0 or another appropriate value
    actors_df.fillna(0, inplace=True)
    
    # Extract features
    feature_columns = [
        'movieCount', 
        'ratingSum', 
        'normalizedMovieRank', 
        'googleHits', 
        'normalizedGoogleRank', 
        'normalizedRating'
    ]
    
    # Handle cases where feature columns might have all zeros or constant values
    # to avoid division by zero during normalization
    features = actors_df[feature_columns].values.astype(float)
    
    # Normalize features
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    features_tensor = torch.tensor(features_normalized, dtype=torch.float32)
    
    # Create adjacency matrix based on cosine similarity
    # Using torch's cosine similarity function
    with torch.no_grad():
        normalized_features = torch.nn.functional.normalize(features_tensor, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    
    # Convert to numpy for further processing if needed
    similarity_matrix = similarity_matrix.numpy()
    
    # Ensure no self-loops by setting diagonal to 0
    np.fill_diagonal(similarity_matrix, 0)
    
    # Convert back to torch tensor
    adjacency_matrix = torch.tensor(similarity_matrix, dtype=torch.float32)
    
    return adjacency_matrix, features_tensor, actors_df

# if __name__ == "__main__":
#     adjacency_matrix, features, actors_df = load_and_preprocess('src/SANDBOX/dataset/BollywoodActorRanking.csv')
#     print("Adjacency Matrix Shape:", adjacency_matrix.shape)
#     print("Features Shape:", features.shape)
#     print("Actors DataFrame Shape:", actors_df.shape)