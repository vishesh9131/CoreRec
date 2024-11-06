import torch
from torch.utils.data import Dataset

from corerec.cr_utility.dataloader import DataLoader
from corerec.engines.contentFilterEngine.nn_based_algorithms import NN__AUTOENCODER

class MoviesDataset(Dataset):
    def __init__(self, file_path):
        self.movies = []
        self.genre_to_idx = {}
        self.idx_to_genre = []
        self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='latin1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) < 3:
                    continue
                movie_id, title, genres = parts
                genre_list = genres.split('|')
                self.movies.append((title, genre_list))
                for genre in genre_list:
                    if genre not in self.genre_to_idx:
                        self.genre_to_idx[genre] = len(self.idx_to_genre)
                        self.idx_to_genre.append(genre)

    def __len__(self):
        return len(self.movies)

    def __getitem__(self, idx):
        title, genre_list = self.movies[idx]
        genre_vector = torch.zeros(len(self.genre_to_idx))
        for genre in genre_list:
            genre_vector[self.genre_to_idx[genre]] = 1
        return genre_vector, title

def train_autoencoder(autoencoder, data_loader, criterion, optimizer, num_epochs):
    autoencoder.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for genre_vector, _ in data_loader:
            optimizer.zero_grad()
            reconstructed = autoencoder(genre_vector)
            loss = criterion(reconstructed, genre_vector)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}")

def main():
    # File path to the movies.dat file
    file_path = 'src/SANDBOX/dataset/ml-1m/movies.dat'
    
    # Initialize Dataset and DataLoader
    dataset = MoviesDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define model parameters
    input_dim = len(dataset.genre_to_idx)
    hidden_dim = 12
    latent_dim = 6
    
    # Initialize Autoencoder
    autoencoder = NN__AUTOENCODER(input_dim, hidden_dim, latent_dim)
    
    # Define Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Train the Autoencoder
    train_autoencoder(autoencoder, dataloader, criterion, optimizer, num_epochs=50)
    
    # Evaluate a sample
    autoencoder.eval()
    with torch.no_grad():
        sample_idx = 0  # Change this index to test different samples
        sample_vector, sample_title = dataset[sample_idx]
        reconstructed_vector = autoencoder(sample_vector.unsqueeze(0)).squeeze()
        
        # Convert vectors to genre names
        original_genres = [dataset.idx_to_genre[i] for i, val in enumerate(sample_vector) if val > 0.5]
        reconstructed_genres = [dataset.idx_to_genre[i] for i, val in enumerate(reconstructed_vector) if val > 0.5]
        
        print(f"Original genres for '{sample_title}': {original_genres}")
        print(f"Reconstructed genres: {reconstructed_genres}")

if __name__ == "__main__":
    main()