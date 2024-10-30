import torch
import pandas as pd
from Tmodel import GraphTransformerV2

# Load the dataset
books_df = pd.read_csv('src/SANDBOX/dataset/books.csv')

# Preprocess the dataset to create a graph
num_books = len(books_df)
adjacency_matrix = torch.eye(num_books)  # Identity matrix as a placeholder
graph_metrics = torch.rand(num_books, num_books)  # Random metrics as a placeholder

# Initialize the model
input_dim = 10000  # Example input dimension
d_model = 8
num_layers = 2
num_heads = 2
d_feedforward = 16
model = GraphTransformerV2(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_feedforward=d_feedforward, input_dim=input_dim)

# Ensure x has the correct number of rows (num_books) and columns (input_dim)
x = torch.rand(num_books, input_dim)

# Train the model (simplified)
output = model(x, adjacency_matrix, graph_metrics)

# Initialize user feedback storage
user_feedback = torch.zeros(num_books)

# Interactive CLI for book recommendation
def recommend_books():
    print("Welcome to the Book Recommendation System!")
    while True:
        user_input = input("Enter a book title you like (or 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break

        # Find the book in the dataset
        book_idx = books_df[books_df['title'].str.contains(user_input, case=False, na=False)].index
        if len(book_idx) == 0:
            print("Book not found. Please try another title.")
            continue

        # Get recommendations based on the model's output
        scores = output[book_idx].detach().numpy()
        recommended_idx = scores.argsort()[0][-5:]  # Get top 5 recommendations

        print("Recommended books:")
        for idx in recommended_idx:
            print(f"- {books_df.iloc[idx]['title']} by {books_df.iloc[idx]['authors']}")
            feedback = input(f"Do you like this recommendation? (1 for Yes, 0 for No): ").strip()
            if feedback == '1':
                user_feedback[idx] += 1  # Increment like count
            elif feedback == '0':
                user_feedback[idx] -= 1  # Decrement like count

        # Optionally, update the model or features based on feedback
        # For example, adjust the feature matrix or retrain the model

recommend_books()

# Example of using feedback to adjust features
def update_features_with_feedback(x, user_feedback):
    # Normalize feedback and add to features
    feedback_normalized = user_feedback / torch.max(torch.abs(user_feedback))
    x += feedback_normalized.unsqueeze(1)  # Add feedback as a new feature dimension
    return x

# Update features with feedback
x = update_features_with_feedback(x, user_feedback)

# Optionally, retrain the model with updated features
# output = model(x, adjacency_matrix, graph_metrics)
