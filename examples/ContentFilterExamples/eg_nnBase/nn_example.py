import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
# Import using the top-level module pattern
import corerec.transformer as transformer
import corerec.rnn as rnn


# Define additional models if necessary
class SourceModel(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(SourceModel, self).__init__()
        self.fc = nn.Linear(input_dim, feature_dim)

    def forward(self, x):
        return self.fc(x)


class TargetModel(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(TargetModel, self).__init__()
        self.fc = nn.Linear(feature_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class MultilingualModel:
    def translate(self, text_input, source_lang, target_lang):
        # Placeholder translation logic
        return f"Translated ({source_lang}->{target_lang}): {text_input}"


class NNBasedRecommender:
    def __init__(self, source_model=None, target_model=None, multilingual_model=None):
        self.title_vectorizer = TfidfVectorizer(max_features=500)
        self.genre_binarizer = MultiLabelBinarizer()

        # Initialize neural network-based algorithms
        self.transformer_model = transformer.TransformerModel(
            input_dim=500,  # Should match the size of your feature vectors
            embed_dim=256,
            num_heads=8,
            hidden_dim=512,
            num_layers=6,
            dropout=0.1,
            num_classes=2,  # Adjust based on your classification task
        )
        self.rnn_model = rnn.RNNModel(
            input_dim=500,  # Should match the size of your feature vectors
            embed_dim=256,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            num_classes=2,  # Adjust based on your classification task
        )

    def load_data(self, filepath):
        self.movies_df = pd.read_csv(
            filepath,
            sep="::",
            engine="python",
            header=None,
            names=["movie_id", "title", "genres"],
            encoding="ISO-8859-1",
        )
        self.movies_df["genres"] = self.movies_df["genres"].apply(lambda x: x.split("|"))
        return self.movies_df

    def preprocess_data(self):
        # Fit the vectorizers on the entire dataset
        title_features = self.title_vectorizer.fit_transform(self.movies_df["title"]).toarray()
        genre_features = self.genre_binarizer.fit_transform(self.movies_df["genres"])
        return title_features, genre_features

    def get_recommendations(self, movie_id, top_k=5):
        # Find the index of the movie in the dataframe
        if movie_id not in self.movies_df["movie_id"].values:
            raise ValueError(f"Movie ID {movie_id} does not exist in the dataset.")

        movie_idx = self.movies_df.index[self.movies_df["movie_id"] == movie_id].tolist()[0]

        # Preprocess data and get all features
        title_features, genre_features = self.preprocess_data()

        # Get the specific movie's features
        target_title = [self.movies_df.at[movie_idx, "title"]]
        target_genres = [self.movies_df.at[movie_idx, "genres"]]
        target_title_features = self.title_vectorizer.transform(target_title).toarray()
        target_genre_features = self.genre_binarizer.transform(target_genres)

        # Calculate cosine similarity
        similarities = cosine_similarity(
            np.hstack((target_title_features, target_genre_features)),
            np.hstack((title_features, genre_features)),
        ).flatten()

        # Get top-k similar movies
        similar_indices = similarities.argsort()[-(top_k + 1) : -1][
            ::-1
        ]  # Exclude the input movie itself
        recommendations = self.movies_df.iloc[similar_indices]
        return recommendations

    def train_transformer(self, data_loader, criterion, optimizer, num_epochs):
        self.transformer_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self.transformer_model(inputs)  # outputs: (batch_size, num_classes)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(data_loader)
            print(f"Transformer Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def train_rnn(self, data_loader, criterion, optimizer, num_epochs):
        self.rnn_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self.rnn_model(
                    inputs
                )  # Ensure inputs have shape (batch_size, seq_length, input_dim)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(data_loader)
            print(f"RNN Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


def main():
    # Initialize models
    source_model = SourceModel(input_dim=500, feature_dim=128)
    target_model = TargetModel(feature_dim=128, output_dim=2)
    multilingual_model = MultilingualModel()

    # Initialize recommender
    recommender = NNBasedRecommender(source_model, target_model, multilingual_model)

    # Load and preprocess data
    print("Loading data...")
    # Try different possible paths
    import os
    possible_paths = [
        "src/SANDBOX/dataset/ml-1m/movies.dat",
        "sample_data/movies.dat",
        "../sample_data/movies.dat",
        "../../sample_data/movies.dat",
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        print("Warning: movies.dat not found. Please provide a valid path.")
        print("Tried paths:", possible_paths)
        return
    
    movies_df = recommender.load_data(file_path)
    print(f"Loaded {len(movies_df)} movies")

    # Generate recommendations
    movie_id = 78  # Example movie ID
    print(f"\nGenerating recommendations for movie ID {movie_id}...")
    recommendations = recommender.get_recommendations(movie_id)
    print("\nTop recommendations:")
    print(recommendations[["title", "genres"]])

    # Initialize Data Loaders, Criterion, and Optimizer
    def get_synthetic_data(input_dim, seq_length, num_samples):
        """
        Generate synthetic data with a sequence dimension.

        Args:
            input_dim (int): Dimension of the input features.
            seq_length (int): Length of the sequence.
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Synthetic input data of shape (num_samples, seq_length, input_dim).
            torch.Tensor: Synthetic labels of shape (num_samples,).
        """
        X = torch.randn(num_samples, seq_length, input_dim)  # (num_samples, seq_length, input_dim)
        y = torch.randint(0, 2, (num_samples,))  # Binary classification
        return X, y

    input_dim = 500  # Should match the output of your feature vectors
    seq_length = 1  # Sequence length set to 1
    num_samples = 1000
    feature_dim = 128
    output_dim = 2  # Example: binary classification

    # Create synthetic data with sequence dimension
    X, y = get_synthetic_data(input_dim, seq_length, num_samples)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=0.001)
    num_epochs = 5

    # Training Transformer
    print("\nTraining Transformer Model...")
    transformer_optimizer = torch.optim.Adam(recommender.transformer_model.parameters(), lr=0.001)
    transformer_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    recommender.train_transformer(
        transformer_data_loader, criterion, transformer_optimizer, num_epochs=3
    )

    # Training RNN
    print("\nTraining RNN Model...")
    rnn_optimizer = torch.optim.Adam(recommender.rnn_model.parameters(), lr=0.001)
    rnn_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    recommender.train_rnn(rnn_data_loader, criterion, rnn_optimizer, num_epochs=3)


if __name__ == "__main__":
    main()
