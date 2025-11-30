# Import using cleaner module path
import corerec as cr

import os
from collections import defaultdict


class RecommenderSystemFilter:
    def __init__(self, movies_dat_path):
        """
        Initializes the RecommenderSystemFilter with the path to the movies.dat dataset.

        Parameters:
        - movies_dat_path (str): The file path to the movies.dat file.
        """
        self.movies_dat_path = movies_dat_path
        self.movie_genres = {}
        self.genre_movies = defaultdict(set)
        self.load_movies()

    def load_movies(self):
        """
        Loads and parses the movies.dat file to build movie-genre mappings.
        """
        if not os.path.exists(self.movies_dat_path):
            raise FileNotFoundError(f"The file {self.movies_dat_path} does not exist.")

        with open(self.movies_dat_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                parts = line.split("::")
                if len(parts) != 3:
                    continue  # Skip malformed lines
                movie_id, title, genres = parts
                genre_list = genres.split("|")
                self.movie_genres[movie_id] = {"title": title, "genres": genre_list}
                for genre in genre_list:
                    self.genre_movies[genre.lower()].add(movie_id)

    def recommend(self, movie_title, top_n=5):
        """
        Recommends movies similar to the given movie title based on genre similarity.

        Parameters:
        - movie_title (str): The title of the movie to base recommendations on.
        - top_n (int): The number of recommendations to return.

        Returns:
        - list: A list of recommended movie titles.
        """
        # Find the movie ID for the given title
        target_movie_id = None
        for movie_id, details in self.movie_genres.items():
            if details["title"].lower() == movie_title.lower():
                target_movie_id = movie_id
                break

        if not target_movie_id:
            raise ValueError(f"Movie titled '{movie_title}' not found in the dataset.")

        target_genres = set(
            [genre.lower() for genre in self.movie_genres[target_movie_id]["genres"]]
        )

        # Find movies that share the most genres with the target movie
        similarity_scores = defaultdict(int)
        for genre in target_genres:
            for movie_id in self.genre_movies[genre]:
                if movie_id != target_movie_id:
                    similarity_scores[movie_id] += 1

        # Sort movies based on similarity scores
        sorted_movies = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top_n recommendations
        recommendations = []
        for movie_id, score in sorted_movies[:top_n]:
            recommendations.append(self.movie_genres[movie_id]["title"])

        return recommendations


def main():
    # Sample Content to Filter
    content = (
        "I absolutely love the new sci-fi movie! The chemistry between characters is fantastic."
    )

    # -----------------------------
    # 1. Rule-Based Filtering
    # -----------------------------
    rule_filter = cr.RuleBased()
    # Adding rules: Flag content containing 'sci-fi' and allow content containing 'fantastic'
    rule_filter.add_rule("sci-fi", "flag")
    rule_filter.add_rule("fantastic", "allow")
    rule_result = rule_filter.filter_content(content)
    print("Rule-Based Filter Result:", rule_result)

    # -----------------------------
    # 2. Sentiment Analysis Filtering
    # -----------------------------
    sentiment_filter = cr.SentimentAnalysis(threshold=0.2)
    sentiment_result = sentiment_filter.filter_content(content)
    print("Sentiment Analysis Filter Result:", sentiment_result)

    # -----------------------------
    # 3. Ontology-Based Filtering
    # -----------------------------
    ontology_path = "src/SANDBOX/contentFilterExample/exampleotheraoprach/ontologies/ontology.owl"
    try:
        ontology_filter = cr.OntologyBased(ontology_path)
        ontology_result = ontology_filter.filter_content(content)
        print("Ontology-Based Filter Result:", ontology_result)
    except ValueError as ve:
        print(ve)

    # -----------------------------
    # 4. Recommender System
    # -----------------------------
    movies_dat_path = "src/SANDBOX/dataset/ml-1m/movies.dat"
    try:
        recommender = RecommenderSystemFilter(movies_dat_path)
        movie_to_recommend = "Toy Story (1995)"
        recommendations = recommender.recommend(movie_to_recommend, top_n=5)
        print(f"Recommendations based on '{movie_to_recommend}':")
        for idx, rec in enumerate(recommendations, start=1):
            print(f"{idx}. {rec}")
    except (FileNotFoundError, ValueError) as e:
        print(e)


if __name__ == "__main__":
    main()
