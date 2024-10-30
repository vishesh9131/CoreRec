# meta_learning implementation
import torch
from corerec.train import train_model
from corerec.predict import predict

class MetaLearner:
    def __init__(self, model, data_loader, criterion, optimizer, num_epochs):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self):
        print("Training with Meta-Learning...")
        train_model(self.model, self.data_loader, self.criterion, self.optimizer, self.num_epochs)

    def predict(self, graph, node_index, top_k=5, threshold=0.5):
        print("Predicting with Meta-Learning...")
        return predict(self.model, graph, node_index, top_k, threshold)
