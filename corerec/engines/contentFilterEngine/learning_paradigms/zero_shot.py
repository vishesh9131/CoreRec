# zero_shot implementation
import torch
from corerec.predict import predict

class ZeroShotLearner:
    def __init__(self, model):
        self.model = model

    def predict(self, graph, node_index, top_k=5, threshold=0.5):
        print("Predicting with Zero-Shot Learning...")
        return predict(self.model, graph, node_index, top_k, threshold)
