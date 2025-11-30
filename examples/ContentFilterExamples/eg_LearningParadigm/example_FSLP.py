import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import corerec as cr

# Learning Paradigms: While this example uses FewShotLearner,
# you can similarly utilize ZeroShotLearner, TransferLearningLearner,
# or MetaLearner by importing and initializing them accordingly.


# Define the neural network model
class MovieGenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MovieGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()  # Using sigmoid for multi-label classification

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# Custom Dataset for Movies
class MoviesDataset(Dataset):
    def __init__(self, filepath, vectorizer=None, mlb=None, fit_vectorizer=True, fit_mlb=True):
        self.data = self.load_data(filepath)
        self.genres = self.extract_genres()

        # Initialize MultiLabelBinarizer for genres
        if mlb is None and fit_mlb:
            self.mlb = MultiLabelBinarizer()
            self.data["genres"] = self.data["genres"].str.split("|")
            self.mlb.fit(self.data["genres"])
        elif mlb is not None:
            self.mlb = mlb
            self.data["genres"] = self.data["genres"].str.split("|")
        else:
            raise ValueError("mlb cannot be None when fit_mlb is False")

        # Initialize TF-IDF Vectorizer for titles
        if vectorizer is None and fit_vectorizer:
            self.vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features
            self.vectorizer.fit(self.data["title"])
        elif vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            raise ValueError("vectorizer cannot be None when fit_vectorizer is False")

        self.features = self.vectorizer.transform(self.data["title"]).toarray()
        self.labels = self.mlb.transform(self.data["genres"])

    def load_data(self, filepath):
        # Load the dataset with specified encoding
        try:
            df = pd.read_csv(
                filepath,
                sep="::",
                engine="python",
                header=None,
                names=["movie_id", "title", "genres"],
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            print("UTF-8 encoding failed. Trying 'latin1' encoding.")
            df = pd.read_csv(
                filepath,
                sep="::",
                engine="python",
                header=None,
                names=["movie_id", "title", "genres"],
                encoding="latin1",
            )
        return df

    def extract_genres(self):
        # Extract unique genres
        genres = set()
        for genre_list in self.data["genres"].str.split("|"):
            genres.update(genre_list)
        return genres

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Features are TF-IDF vectors of titles
        features = self.features[idx]
        labels = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )


def main():
    # Parameters
    # Try different possible paths
    import os
    possible_paths = [
        "src/SANDBOX/dataset/ml-1m/movies.dat",
        "sample_data/movies.dat",
        "../sample_data/movies.dat",
        "../../sample_data/movies.dat",
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("Warning: movies.dat not found. Please provide a valid path.")
        print("Tried paths:", possible_paths)
        return
    hidden_size = 128  # Increased hidden size for more complexity
    num_epochs = 20  # Increased number of epochs
    batch_size = 32
    learning_rate = 0.0005  # Reduced learning rate

    # Initialize dataset and dataloader
    # First, create a dataset to fit the vectorizer and binarizer
    initial_dataset = MoviesDataset(filepath=dataset_path)

    # Extract vectorizer and multilabel binarizer from the initial dataset
    vectorizer = initial_dataset.vectorizer
    mlb = initial_dataset.mlb

    # Create the final dataset using the fitted vectorizer and binarizer
    dataset = MoviesDataset(
        filepath=dataset_path, vectorizer=vectorizer, mlb=mlb, fit_vectorizer=False, fit_mlb=False
    )

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get input size and number of classes from the dataset
    input_size = dataset.features.shape[1]
    num_classes = dataset.labels.shape[1]

    # Initialize the model, loss function, and optimizer
    model = MovieGenreClassifier(input_size, hidden_size, num_classes)
    criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the Few-Shot Learner
    # Note: FewShotLearner is currently commented out, using TransferLearningLearner as alternative
    learner = cr.TransferLearning(model, data_loader, criterion, optimizer, num_epochs)

    # Train the model
    learner.train()

    # Example prediction for all movies
    model.eval()
    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample_features, _ = dataset[idx]
            sample_features = sample_features.unsqueeze(0)  # Add batch dimension
            output = model(sample_features)
            predicted_genres = output.squeeze().detach().cpu().numpy()
            # Threshold to decide if a genre is predicted
            threshold = 0.5
            recommended_genres = [
                genre
                for genre, idx_genre in zip(mlb.classes_, range(num_classes))
                if predicted_genres[idx_genre] > threshold
            ]
            movie_title = dataset.data.iloc[idx]["title"]
            results.append({"title": movie_title, "recommended_genres": recommended_genres})
            print(f"Recommended Genres for '{movie_title}': {recommended_genres}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("src/SANDBOX/contentFilterExample/vishesh_results.csv", index=False)


if __name__ == "__main__":
    main()
