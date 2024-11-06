# zero_shot implementation
import torch
from corerec.predict import predict
class ZeroShotLearner:
    def __init__(self, model):
        """
        Initializes the ZeroShotLearner with the given model.

        Parameters:
            model (torch.nn.Module): The model to be used for predictions.
        """
        self.model = model

    def predict(self, graph, node_index, top_k=5, threshold=0.5):
        """
        Predicts the top-k items for a given node in a graph using the model.

        Parameters:
            graph (torch.Tensor): The graph data structure containing nodes and edges.
            node_index (int): The index of the node for which predictions are to be made.
            top_k (int, optional): The number of top items to return. Defaults to 5.
            threshold (float, optional): The threshold for prediction confidence. Defaults to 0.5.

        Returns:
            List[int]: A list of indices representing the top-k predicted items.
        """
        print("Predicting with Zero-Shot Learning...")
        return predict(self.model, graph, node_index, top_k, threshold)
