import pandas as pd
import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.triples import TriplesFactory

# Load the labels
labels = pd.read_csv('labelele.csv')['Names'].tolist()

# Create a mapping from labels to indices
label_to_index = {label.strip(): idx for idx, label in enumerate(labels)}

# Print the label_to_index dictionary to verify
print(label_to_index)

# Load the adjacency matrix
adj_matrix = np.loadtxt('label.csv', delimiter=',')

# Convert adjacency matrix to triples
triples = []

for i in range(len(labels)):
    for j in range(len(labels)):
        if adj_matrix[i, j] != 0:  # Only consider non-zero relationships
            triples.append([labels[i].strip(), 'connected_to', labels[j].strip()])

triples = np.array(triples)

# Create a TriplesFactory for the triples
triples_factory = TriplesFactory.from_labeled_triples(triples)

# Split into training and testing sets
training_factory, testing_factory = triples_factory.split([0.8, 0.2])

# Train the TransE model using PyKEEN
result = pipeline(
    training=training_factory,
    testing=testing_factory,
    model='TransE',
    model_kwargs=dict(embedding_dim=100),
    training_kwargs=dict(num_epochs=200),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.0001),
    regularizer='LP',
    regularizer_kwargs=dict(p=3, weight=0.0001),
)

# Get the trained model
model = result.model

# Make recommendations
node = 'vishesh'.strip()
relation = 'connected_to'

# Create a list of all possible triples with the given node and relation, excluding self-referential triples
possible_triples = np.array([[node, relation, target.strip()] for target in labels if target.strip() != node])

# Convert the possible triples to integer indices
possible_triples_indices = np.array([[label_to_index[h], 0, label_to_index[t]] for h, r, t in possible_triples])

# Debug print to verify the shape of possible_triples_indices
print("Shape of possible_triples_indices:", possible_triples_indices.shape)

# Convert to PyTorch tensor
possible_triples_tensor = torch.tensor(possible_triples_indices, dtype=torch.long)

# Get scores for all possible triples
scores = model.predict_hrt(possible_triples_tensor)

# Detach the tensor and convert to NumPy array
scores_np = scores.detach().numpy()

# Get the index of the top score
top_index = np.argmax(scores_np)

# Debug print to verify the top index
print("Top index:", top_index)

# Get the top similar node
top_triple = possible_triples_indices[top_index]

# Debug print to verify the top triple
print("Top triple:", top_triple)

# Get the top similar node
similar_node = labels[top_triple[4]]

print(f'Top similar node to {node}: {similar_node}')