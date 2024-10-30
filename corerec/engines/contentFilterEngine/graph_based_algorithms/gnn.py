# gnn implementation
import networkx as nx
import matplotlib.pyplot as plt
from corerec.vish_graphs import draw_graph

class GNN:
    def __init__(self):
        self.graph = None

    def load_graph(self, file_path):
        """
        Load a graph from a file.
        """
        self.graph = nx.read_adjlist(file_path, delimiter=',', nodetype=int)
        return self.graph

    def visualize_graph(self, recommended_nodes=None, top_nodes=None, node_labels=None):
        """
        Visualize the graph using the draw_graph function.
        """
        if self.graph is not None:
            # Use a simpler layout for visualization
            pos = nx.circular_layout(self.graph)
            draw_graph(self.graph, pos=pos, 
                       top_nodes=top_nodes, 
                       recommended_nodes=recommended_nodes, 
                       node_labels=node_labels)
        else:
            print("Graph not loaded. Please load a graph first.")

# Example usage
# gnn = GNN()
# gnn.load_graph('graph_dataset.csv')
# gnn.visualize_graph()
