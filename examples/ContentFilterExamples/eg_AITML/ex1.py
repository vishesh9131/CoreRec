# Implementation of AITM-based Recommender System using movies.dat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from corerec.engines.contentFilterEngine.nn_based_algorithms.AITM import AITM

# Define SourceModel and TargetModel
class SourceModel(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(SourceModel, self).__init__()
        self.fc = nn.Linear(input_dim, feature_dim)
    
    def forward(self, x):
        return torch.relu(self.fc(x))

class TargetModel(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(TargetModel, self).__init__()
        self.fc = nn.Linear(feature_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

class MultilingualModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultilingualModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        return self.fc2(out)

# Custom Dataset for Movies
class MoviesDataset(Dataset):
    def __init__(self, filepath, encoding='latin1'):
        self.movies = self.load_movies(filepath, encoding)
        self.genres = self.extract_genres(self.movies)
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(set([g for sublist in self.genres for g in sublist])))}
        self.encoded_genres = self.encode_genres(self.genres)
    
    def load_movies(self, filepath, encoding):
        movies = []
        try:
            with open(filepath, 'r', encoding=encoding) as file:
                for line in file:
                    if line.strip() and '::' in line:
                        parts = line.strip().split('::')
                        if len(parts) == 3:
                            movie_id, title, genres = parts
                            movies.append({'movie_id': int(movie_id), 'title': title, 'genres': genres.split('|')})
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError: {e}")
            print("Please verify the file encoding.")
            raise e
        return movies
    
    def extract_genres(self, movies):
        return [movie['genres'] for movie in movies]
    
    def encode_genres(self, genres_list):
        encoded = []
        for genres in genres_list:
            genre_vec = np.zeros(len(self.genre_to_idx))
            for genre in genres:
                if genre in self.genre_to_idx:
                    genre_vec[self.genre_to_idx[genre]] = 1
            encoded.append(genre_vec)
        return np.array(encoded, dtype=np.float32)
    
    def __len__(self):
        return len(self.movies)
    
    def __getitem__(self, idx):
        # For simplicity, using genres as both input and target (autoencoder-like)
        input_genre = self.encoded_genres[idx]
        target_genre = self.encoded_genres[idx]
        return torch.tensor(input_genre), torch.tensor(target_genre)

def recommend_genres_for_movie(movie_title, dataset, model):
    # Find the movie in the dataset
    movie_idx = None
    for idx, movie in enumerate(dataset.movies):
        if movie['title'] == movie_title:
            movie_idx = idx
            break
    
    if movie_idx is None:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return
    
    # Prepare the input
    input_genre, _ = dataset[movie_idx]
    input_genre = input_genre.unsqueeze(0)  # Add batch dimension
    
    # Make predictions
    model.source_model.eval()
    with torch.no_grad():
        source_output = model.source_model(input_genre)
        predicted_genres = model.target_model(source_output).squeeze()
    
    # Decode the output
    idx_to_genre = {idx: genre for genre, idx in dataset.genre_to_idx.items()}
    recommended_genres = [idx_to_genre[i] for i, val in enumerate(predicted_genres) if val > 0.5]
    
    print(f"Recommended Genres for '{movie_title}': {recommended_genres}")

def main():
    # Parameters
    input_dim = 18  # Adjust this to match the actual number of unique genres
    feature_dim = 50
    output_dim = 18  # This should also match the number of unique genres
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Initialize Dataset and DataLoader
    dataset = MoviesDataset('src/SANDBOX/dataset/ml-1m/movies.dat', encoding='latin1')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize AITM
    aitm = AITM(input_dim, feature_dim, output_dim)
    
    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(aitm.parameters(), lr=learning_rate)
    
    # Training Loop
    print("Starting Training...")
    aitm.train(dataloader, criterion, optimizer, num_epochs)
    
    # Evaluation
    print("Evaluating Model...")
    aitm.evaluate(dataloader, criterion)
    
    # Recommend genres for a specific movie
    movie_title = "Mr. Smith Goes to Washington (1939)"
    recommend_genres_for_movie(movie_title, dataset, aitm)

if __name__ == "__main__":
    main()
