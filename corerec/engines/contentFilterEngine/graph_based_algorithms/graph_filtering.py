# graph_filtering implementation
from corerec.vish_graphs import (
    generate_random_graph,
    generate_large_random_graph,
    scale_and_save_matrices
)

class GraphFiltering:
    def __init__(self):
        self.graph = None

    def generate_graph(self, num_people, file_path="graph_dataset.csv", seed=None):
        """
        Generate a random graph and save it to a file.
        """
        self.graph = generate_random_graph(num_people, file_path, seed)
        return self.graph

    def generate_large_graph(self, num_people, file_path="large_random_graph.csv", seed=None):
        """
        Generate a large random graph using multiprocessing.
        """
        self.graph = generate_large_random_graph(num_people, file_path, seed)
        return self.graph

    def scale_and_save(self, input_file, output_dir, num_matrices):
        """
        Scale and save matrices to the specified directory.
        """
        scale_and_save_matrices(input_file, output_dir, num_matrices)

# Example usage
# graph_filter = GraphFiltering()
# graph_filter.generate_graph(100)
# graph_filter.scale_and_save('SANDBOX/label.csv', 'SANDBOX/delete', 10)
