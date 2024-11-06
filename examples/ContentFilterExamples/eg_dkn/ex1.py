import torch
from torch.utils.data import DataLoader
from corerec.engines.contentFilterEngine.nn_based_algorithms.dkn import DKN
from corerec.engines.contentFilterEngine.nn_based_algorithms.autoencoder import Autoencoder  # Example import

class MoviesDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, vocab, entity_to_idx):
        self.movies = []
        self.genre_to_idx = {}
        self.idx_to_genre = []
        self.vocab = vocab
        self.entity_to_idx = entity_to_idx
        self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='latin1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) < 3:
                    continue
                movie_id, title, genres = parts
                text, entities = self.process_title(title)
                genre_list = genres.split('|')
                self.movies.append((text, entities, genre_list))
                for genre in genre_list:
                    if genre not in self.genre_to_idx:
                        self.genre_to_idx[genre] = len(self.idx_to_genre)
                        self.idx_to_genre.append(genre)

    def process_title(self, title):
        tokens = title.lower().split()
        text = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        entities = [self.entity_to_idx.get(token, 0) for token in tokens]
        return text, entities

    def __len__(self):
        return len(self.movies)

    def __getitem__(self, idx):
        text, entities, genre_list = self.movies[idx]
        genre_vector = torch.zeros(len(self.genre_to_idx))
        for genre in genre_list:
            genre_vector[self.genre_to_idx[genre]] = 1
        return torch.tensor(text, dtype=torch.long), torch.tensor(entities, dtype=torch.long), genre_vector

def build_vocab(file_path):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    with open(file_path, 'r', encoding='latin1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) < 3:
                continue
            movie_id, title, genres = parts
            tokens = title.lower().split()
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab

def build_entity_to_idx(file_path):
    entity_to_idx = {}
    with open(file_path, 'r', encoding='latin1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) < 3:
                continue
            movie_id, title, genres = parts
            tokens = title.lower().split()
            for token in tokens:
                if token not in entity_to_idx:
                    entity_to_idx[token] = len(entity_to_idx) + 1  # Reserve 0 for unknown
    return entity_to_idx

def collate_fn(batch):
    texts, entities, genres = zip(*batch)
    # Pad texts
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    # Pad entities
    padded_entities = torch.nn.utils.rnn.pad_sequence(entities, batch_first=True, padding_value=0)
    genres = torch.stack(genres)
    return padded_texts, padded_entities, genres

def main():
    # File path to the movies.dat file
    file_path = 'src/SANDBOX/dataset/ml-1m/movies.dat'
    
    # Build Vocabulary and Entity Dictionary
    vocab = build_vocab(file_path)
    entity_to_idx = build_entity_to_idx(file_path)
    
    # Initialize Dataset and DataLoader
    dataset = MoviesDataset(file_path, vocab, entity_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model parameters
    vocab_size = len(vocab)
    embedding_dim = 128
    entity_embedding_dim = 128
    knowledge_graph_size = len(entity_to_idx) + 1  # +1 for padding/unknown
    
    # Initialize DKN model
    dkn_model = DKN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        entity_embedding_dim=entity_embedding_dim,
        knowledge_graph_size=knowledge_graph_size,
        text_kernel_sizes=[3, 4, 5],
        text_num_filters=100,
        dropout=0.5,
        num_classes=len(dataset.genre_to_idx)  # Multi-label classification
    ).to(device)
    
    # Define Loss and Optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(dkn_model.parameters(), lr=0.001)
    
    # Training Loop
    dkn_model.train()
    for epoch in range(10):  # Example: 10 epochs
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            texts, entities, genres = batch
            texts = texts.to(device)
            entities = entities.to(device)
            genres = genres.to(device)
            
            optimizer.zero_grad()
            outputs = dkn_model(texts, entities)
            loss = criterion(outputs, genres)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(dkn_model.state_dict(), 'dkn_model.pth')
    print("Model saved to dkn_model.pth")

if __name__ == "__main__":
    main()
