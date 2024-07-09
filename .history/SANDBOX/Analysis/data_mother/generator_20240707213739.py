import csv
import random
import numpy as np

# List of meaningful labels
LABEL_CATEGORIES = [
    "Topic", "Genre", "Emotion", "Location", "Industry", "Technology", "Event",
    "Skill", "Product", "Service", "Demographics", "Activity", "Concept"
]

LABEL_DESCRIPTORS = [
    "Data Science", "Machine Learning", "Artificial Intelligence", "Cloud Computing",
    "Cybersecurity", "Blockchain", "Internet of Things", "Augmented Reality",
    "Virtual Reality", "Robotics", "5G", "Quantum Computing", "Edge Computing",
    "Big Data", "DevOps", "Bioinformatics", "Natural Language Processing",
    "Computer Vision", "Autonomous Vehicles", "Renewable Energy", "Space Technology",
    "Nanotechnology", "Biotechnology", "Fintech", "E-commerce", "Social Media",
    "Digital Marketing", "User Experience", "Mobile Development", "Web Development",
    "Game Development", "3D Printing", "Drones", "Wearable Technology", "Smart Home"
]

# Generate labelled.csv with meaningful labels
def generate_labels(num_labels=500):
    labels = []
    for _ in range(num_labels):
        category = random.choice(LABEL_CATEGORIES)
        descriptor = random.choice(LABEL_DESCRIPTORS)
        label = f"{category}_{descriptor}"
        labels.append(label)
    
    # Ensure uniqueness
    labels = list(set(labels))
    
    # If we don't have enough unique labels, add numbered labels
    while len(labels) < num_labels:
        labels.append(f"Custom_Label_{len(labels) + 1}")
    
    # Shuffle the labels
    random.shuffle(labels)
    
    with open('SANDBOX/Analysis/data_mother/500label.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Names'])  # Header
        for label in labels[:num_labels]:
            writer.writerow([label])
    
    print(f"5000label.csv has been generated with {num_labels} meaningful labels.")
    return labels[:num_labels]

# Generate label.csv (adjacency matrix)
def generate_adjacency_matrix(labels):
    num_nodes = len(labels)
    
    # Create a random adjacency matrix
    matrix = np.random.rand(num_nodes, num_nodes)
    
    # Make it symmetric (for undirected graph)
    matrix = (matrix + matrix.T) / 2
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(matrix, 0)
    
    # Round to 2 decimal places
    matrix = np.round(matrix, 2)
    
    # Save to CSV with labels
    with open('SANDBOX/Analysis/data_mother/wgtlabel.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([''])  # Header row
        for i, row in enumerate(matrix):
            writer.writerow(list(row))
    
    print(f"wgtlabel.csv (adjacency matrix) has been generated for {num_nodes} nodes.")

# Generate both datasets
def generate_datasets(num_labels=5000):
    labels = generate_labels(num_labels)
    generate_adjacency_matrix(labels)
    print("Both datasets have been generated successfully.")

# Run the generator
generate_datasets()


