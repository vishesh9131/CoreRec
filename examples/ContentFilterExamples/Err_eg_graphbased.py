import pandas as pd
import networkx as nx
from corerec.engines.contentFilterEngine.graph_based_algorithms import (
    GRA_GRAPH_FILTERING,
    GRA_GNN,
    GRA_SEMANTIC_MODELS
)

def load_movies(file_path: str) -> pd.DataFrame:
    """
    Load and parse the movies.dat file.

    Parameters:
    - file_path (str): Path to the movies.dat file.

    Returns:
    - pd.DataFrame: DataFrame containing movie information.
    """
    column_names = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=column_names,
        encoding='latin-1'
    )
    return movies

def build_genre_graph(movies_df: pd.DataFrame) -> nx.Graph:
    """
    Build a graph where nodes are movies and edges represent shared genres.

    Parameters:
    - movies_df (pd.DataFrame): DataFrame containing movie information.

    Returns:
    - nx.Graph: A graph with movies as nodes and shared genres as edges.
    """
    G = nx.Graph()
    genre_to_movies = {}

    # Create a mapping from genres to movies
    for _, row in movies_df.iterrows():
        movie_id = row['movie_id']
        genres = row['genres'].split('|')
        G.add_node(movie_id, title=row['title'], genres=genres)
        for genre in genres:
            if genre not in genre_to_movies:
                genre_to_movies[genre] = []
            genre_to_movies[genre].append(movie_id)

    # Add edges between movies that share genres
    for genre, movie_ids in genre_to_movies.items():
        for i in range(len(movie_ids)):
            for j in range(i + 1, len(movie_ids)):
                G.add_edge(movie_ids[i], movie_ids[j])

    return G

def main():
    # Load movies data
    movies_file = 'src/SANDBOX/dataset/ml-1m/movies.dat'
    movies_df = load_movies(movies_file)

    # Build genre graph
    genre_graph = build_genre_graph(movies_df)

    # Initialize graph-based algorithms
    graph_filter = GRA_GRAPH_FILTERING()
    gnn = GRA_GNN()
    semantic_models = GRA_SEMANTIC_MODELS()

    # Load the graph into GNN
    gnn.graph = genre_graph

    # Visualize the graph
    # gnn.visualize_graph()

    # Set the graph for semantic models
    semantic_models.set_graph(genre_graph)

    # Find the optimal path
    semantic_models.find_optimal_path(0)

if __name__ == "__main__":
    main()
