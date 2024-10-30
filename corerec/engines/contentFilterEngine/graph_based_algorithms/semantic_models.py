from corerec.vish_graphs import run_optimal_path

class SemanticModels:
    def __init__(self):
        self.graph = []

    def set_graph(self, graph):
        """
        Set the graph for semantic models.
        """
        self.graph = graph

    def find_optimal_path(self, start_city):
        """
        Find the optimal path using the run_optimal_path function.
        """
        run_optimal_path(self.graph, start_city)

# Example usage
# semantic_model = SemanticModels()
# semantic_model.set_graph(graph)
# semantic_model.find_optimal_path(0)
