import torch
from torch.utils.data import Dataset, DataLoader
from corerec.cr_utility.dataloader import DataLoader as CRDataLoader
from corerec.engines.contentFilterEngine.nn_based_algorithms import NN__CNN

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

def train_model(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for genre_vector, _ in data_loader:
            genre_vector = genre_vector.to(device)
            optimizer.zero_grad()
            outputs = model(genre_vector)
            loss = criterion(outputs, genre_vector)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def evaluate_model(model, dataset, idx_to_genre, sample_idx=0, threshold=0.5, device='cpu'):
    model.eval()
    with torch.no_grad():
        sample_vector, sample_title = dataset[sample_idx]
        sample_vector = sample_vector.to(device)
        reconstructed_vector = model(sample_vector.unsqueeze(0)).squeeze()
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(reconstructed_vector).cpu().numpy()
        
        # Convert vectors to genre names based on threshold
        original_genres = [idx_to_genre[i] for i, val in enumerate(sample_vector.cpu().numpy()) if val > 0.5]
        reconstructed_genres = [idx_to_genre[i] for i, val in enumerate(probabilities) if val > threshold]
        
        print(f"Original genres for '{sample_title}': {original_genres}")
        print(f"Reconstructed genres: {reconstructed_genres}")

def main():
    # File path to the movies.dat file
    file_path = 'src/SANDBOX/dataset/ml-1m/movies.dat'
    
    # Initialize Dataset and DataLoader
    dataset = MoviesDataset(file_path)
    dataloader = CRDataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model parameters
    input_dim = len(dataset.genre_to_idx)
    num_classes = input_dim  # For multi-label classification
    emb_dim = 128
    kernel_sizes = [3, 4, 5]
    num_filters = 100
    dropout = 0.5
    num_epochs = 50
    
    # Initialize CNN model
    cnn_model = NN__CNN(
        input_dim=input_dim,
        num_classes=num_classes,
        emb_dim=emb_dim,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters,
        dropout=dropout
    ).to(device)
    
    # Define Loss and Optimizer
    criterion = torch.nn.BCEWithLogitsLoss()  # Suitable for multi-label classification
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    
    # Train the CNN model
    train_model(cnn_model, dataloader, criterion, optimizer, num_epochs, device)
    
    # Evaluate a sample
    evaluate_model(cnn_model, dataset, dataset.idx_to_genre, sample_idx=0, threshold=0.5, device=device)

if __name__ == "__main__":
    main()
