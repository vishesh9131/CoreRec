import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import corerec as cr
from torch.utils.data import Dataset


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
    hidden_size = 256  # Increased hidden size
    num_epochs = 30
    batch_size = 64
    learning_rate = 0.001
    val_split = 0.2

    # Initialize dataset
    dataset = MoviesDataset(filepath=dataset_path)

    # Split dataset into train and validation
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_size = dataset.features.shape[1]
    num_classes = dataset.labels.shape[1]
    model = MovieGenreClassifier(input_size, hidden_size, num_classes)

    # Try to load pre-trained weights
    try:
        model.load_state_dict(torch.load("pretrained_weights.pth"))
        print("Loaded pre-trained weights successfully")
    except:
        print("No pre-trained weights found, starting from scratch")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the Transfer Learning Learner
    learner = cr.TransferLearning(model, train_loader, criterion, optimizer, num_epochs)

    # Train using the learner's train method
    learner.train()

    # Make predictions
    model.eval()
    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample_features, true_labels = dataset[idx]
            sample_features = sample_features.unsqueeze(0)

            # Use the learner's predict method
            predicted_probs = model(sample_features).squeeze().detach().cpu().numpy()

            threshold = 0.5  # Increased threshold for more confident predictions
            recommended_genres = [
                genre
                for genre, idx_genre in zip(dataset.mlb.classes_, range(len(dataset.mlb.classes_)))
                if predicted_probs[idx_genre] > threshold
            ]

            movie_title = dataset.data.iloc[idx]["title"]
            results.append(
                {
                    "title": movie_title,
                    "recommended_genres": recommended_genres,
                    "true_genres": [
                        genre
                        for genre, is_true in zip(dataset.mlb.classes_, true_labels)
                        if is_true
                    ],
                }
            )

            if idx < 5:  # Print first 5 predictions for debugging
                print(f"\nMovie: {movie_title}")
                print(f"True Genres: {results[-1]['true_genres']}")
                print(f"Predicted Genres: {recommended_genres}")
                print(f"Prediction Probabilities: {predicted_probs}")

    # Calculate accuracy metrics
    correct_predictions = 0
    total_predictions = 0
    for result in results:
        true_set = set(result["true_genres"])
        pred_set = set(result["recommended_genres"])
        correct_predictions += len(true_set.intersection(pred_set))
        total_predictions += len(true_set)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "src/SANDBOX/contentFilterExample/LP/transfer_learning_results.csv", index=False
    )


if __name__ == "__main__":
    main()
