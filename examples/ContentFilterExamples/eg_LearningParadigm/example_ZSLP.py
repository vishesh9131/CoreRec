import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import corerec as cr


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
    hidden_size = 128
    input_size = 1000  # Match with TfidfVectorizer max_features
    num_classes = 18  # Typical number of movie genres

    # Initialize dataset just for feature extraction and prediction
    dataset = MoviesDataset(filepath=dataset_path)

    # Initialize the model
    model = MovieGenreClassifier(input_size, hidden_size, num_classes)

    # Load pre-trained weights here if available
    # model.load_state_dict(torch.load('pretrained_weights.pth'))

    # Initialize the Zero-Shot Learner
    learner = cr.ZeroShot(model)

    # Example prediction for all movies
    model.eval()
    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample_features, _ = dataset[idx]
            sample_features = sample_features.unsqueeze(0)

            # Use the zero-shot learner for prediction
            # Note: Since the original zero-shot implementation expects a graph,
            # we'll need to modify the prediction call
            output = model(sample_features)
            predicted_genres = output.squeeze().detach().cpu().numpy()

            threshold = 0.5
            recommended_genres = [
                genre
                for genre, idx_genre in zip(dataset.mlb.classes_, range(len(dataset.mlb.classes_)))
                if predicted_genres[idx_genre] > threshold
            ]
            movie_title = dataset.data.iloc[idx]["title"]
            results.append({"title": movie_title, "recommended_genres": recommended_genres})
            print(f"Zero-Shot Predictions for '{movie_title}': {recommended_genres}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("src/SANDBOX/contentFilterExample/LP/zeroshot_results.csv", index=False)


if __name__ == "__main__":
    main()
