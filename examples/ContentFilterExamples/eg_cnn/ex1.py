import torch
from torch.utils.data import Dataset, DataLoader
from corerec.cr_utility.dataloader import DataLoader as CRDataLoader
from corerec.engines.contentFilterEngine.nn_based_algorithms import NN__CNN
import math

# Set device with proper MPS support
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.backends.mps.enable_fallback_to_cpu = False
        return 'mps'
    return 'cpu'

device = get_device()
print(f"Using device: {device}")

class MoviesDataset(Dataset):
    def __init__(self, file_path):
        self.movies = []
        self.genre_to_idx = {}
        self.idx_to_genre = []
        self.device = get_device()
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
        genre_vector = torch.zeros(len(self.genre_to_idx), device=self.device)
        for genre in genre_list:
            genre_vector[self.genre_to_idx[genre]] = 1
        return genre_vector, title

def train_model(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = torch.tensor(0.0, device=device)
        for genre_vector, _ in data_loader:
            genre_vector = genre_vector.to(device)
            optimizer.zero_grad()
            outputs = model(genre_vector)
            loss = criterion(outputs, genre_vector)
            loss.backward()
            optimizer.step()
            total_loss += loss
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss.item():.4f}")

def calculate_metrics(predictions, ground_truth, k=10):
    """
    Calculate HR@k and NDCG@k metrics.
    
    Args:
        predictions: torch.Tensor of predicted scores
        ground_truth: torch.Tensor of true labels
        k: int, cutoff for metrics
    
    Returns:
        hr: float, Hit Ratio @ k
        ndcg: float, Normalized Discounted Cumulative Gain @ k
    """
    # Ensure k is not larger than the number of items
    k = min(k, len(predictions))
    
    # Get top k predictions
    _, top_k_indices = torch.topk(predictions, k)
    
    # Calculate Hit Ratio
    hit_tensor = torch.zeros_like(ground_truth)
    hit_tensor[top_k_indices] = 1
    hit = torch.any(torch.logical_and(hit_tensor, ground_truth)).float()
    
    # Calculate NDCG
    dcg = torch.tensor(0.0, device=predictions.device)
    idcg = torch.tensor(0.0, device=predictions.device)
    
    # Calculate DCG
    for i, idx in enumerate(top_k_indices):
        if ground_truth[idx] == 1:
            dcg += 1.0 / math.log2(i + 2)
    
    # Calculate IDCG
    n_relevant = int(torch.sum(ground_truth).item())  # Convert to integer
    for i in range(min(n_relevant, k)):
        idcg += 1.0 / math.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else torch.tensor(0.0, device=predictions.device)
    
    return hit.item(), ndcg.item()

def evaluate_model(model, dataset, idx_to_genre, sample_idx=0, threshold=0.5, k=10, device=device):
    model.eval()
    with torch.no_grad():
        # Get sample data
        sample_vector, sample_title = dataset[sample_idx]
        sample_vector = sample_vector.to(device)
        reconstructed_vector = model(sample_vector.unsqueeze(0)).squeeze()
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(reconstructed_vector)
        
        # Calculate metrics with k value check
        max_k = len(probabilities)
        if k > max_k:
            print(f"\nWarning: Requested k={k} is larger than number of items ({max_k}). Using k={max_k}")
            k = max_k
            
        hr, ndcg = calculate_metrics(probabilities, sample_vector, k=k)
        
        # Keep everything on device
        threshold_tensor = torch.tensor(threshold, device=device)
        
        # Get indices where values exceed threshold
        original_indices = torch.nonzero(sample_vector > 0.5).flatten()
        reconstructed_indices = torch.nonzero(probabilities > threshold_tensor).flatten()
        
        # Convert indices to genres (keeping on device)
        original_genres = [idx_to_genre[idx.item()] for idx in original_indices]
        reconstructed_genres = [idx_to_genre[idx.item()] for idx in reconstructed_indices]
        
        print(f"\nEvaluation Results for '{sample_title}':")
        print(f"Original genres: {original_genres}")
        print(f"Reconstructed genres: {reconstructed_genres}")
        print(f"Hit Ratio@{k}: {hr:.4f}")
        print(f"NDCG@{k}: {ndcg:.4f}")
        
        return hr, ndcg

def evaluate_all(model, dataset, idx_to_genre, k=10, device=device):
    """
    Evaluate the model on all samples in the dataset.
    """
    total_hr = torch.tensor(0.0, device=device)
    total_ndcg = torch.tensor(0.0, device=device)
    n_samples = len(dataset)
    
    print("\nEvaluating model on all samples...")
    for idx in range(n_samples):
        sample_vector, _ = dataset[idx]
        sample_vector = sample_vector.to(device)
        
        with torch.no_grad():
            reconstructed_vector = model(sample_vector.unsqueeze(0)).squeeze()
            probabilities = torch.sigmoid(reconstructed_vector)
            hr, ndcg = calculate_metrics(probabilities, sample_vector, k=k)
            total_hr += hr
            total_ndcg += ndcg
    
    avg_hr = total_hr / n_samples
    avg_ndcg = total_ndcg / n_samples
    
    print(f"\nOverall Evaluation Results:")
    print(f"Average HR@{k}: {avg_hr:.4f}")
    print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
    
    return avg_hr, avg_ndcg

def main():
    # File path to the movies.dat file
    file_path = 'src/SANDBOX/dataset/ml-1m/movies.dat'
    
    # Initialize Dataset and DataLoader
    dataset = MoviesDataset(file_path)
    dataloader = CRDataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define model parameters
    input_dim = len(dataset.genre_to_idx)
    num_classes = input_dim
    emb_dim = 128
    kernel_sizes = [3, 4, 5]
    num_filters = 100
    dropout = 0.5
    num_epochs = 200
    
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
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    
    # Train the CNN model
    train_model(cnn_model, dataloader, criterion, optimizer, num_epochs, device)
    
    # Evaluate sample and calculate metrics with reasonable k values
    k_values = [5, 10, min(20, len(dataset.genre_to_idx))]
    print("\nEvaluating model with different k values...")
    
    for k in k_values:
        print(f"\nMetrics for k={k}:")
        hr, ndcg = evaluate_model(cnn_model, dataset, dataset.idx_to_genre, 
                                sample_idx=12, threshold=0.5, k=k, device=device)
    
    # Evaluate on all samples with a safe k value
    print("\nCalculating overall metrics...")
    k = min(10, len(dataset.genre_to_idx))
    avg_hr, avg_ndcg = evaluate_all(cnn_model, dataset, dataset.idx_to_genre, 
                                  k=k, device=device)

if __name__ == "__main__":
    main()
