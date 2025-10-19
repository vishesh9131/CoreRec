# transfer_learning implementation
import torch
from corerec.train import train_model
from corerec.predict import predict

class TransferLearningLearner:
    def __init__(self, model, data_loader, criterion, optimizer, num_epochs):
        """
        Initializes the TransferLearningLearner with the given model, data loader, criterion, optimizer, and number of epochs.

        Parameters:
            model (torch.nn.Module): The model to be trained and used for predictions.
            data_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
            criterion (torch.nn.Module): Loss function used to evaluate the model's performance.
            optimizer (torch.optim.Optimizer): Optimization algorithm used to update model weights.
            num_epochs (int): Number of epochs to train the model.
        """
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self):
        """
        Trains the model using the transfer learning approach.

        This method iterates over the data provided by the data_loader for a specified number of epochs,
        updating the model's weights using the optimizer and evaluating its performance using the criterion.
        """
        print("Training with Transfer Learning...")
        train_model(self.model, self.data_loader, self.criterion, self.optimizer, self.num_epochs)

    def predict(self, graph, node_index, top_k=5, threshold=0.5):
        """
        Predicts the top-k items for a given node in a graph using the trained model.

        Parameters:
            graph (torch.Tensor): The graph data structure containing nodes and edges.
            node_index (int): The index of the node for which predictions are to be made.
            top_k (int, optional): The number of top items to return. Defaults to 5.
            threshold (float, optional): The threshold for prediction confidence. Defaults to 0.5.

        Returns:
            List[int]: A list of indices representing the top-k predicted items.
        """
        print("Predicting with Transfer Learning...")
        return predict(self.model, graph, node_index, top_k, threshold)
