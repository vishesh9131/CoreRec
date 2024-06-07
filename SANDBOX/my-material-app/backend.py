from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
import core_rec as cs
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/labels', methods=['GET'])
def get_labels():
    try:
        print("Received request for labels")
        file_path = os.path.join(BASE_DIR, "ML", "Data", "labelele.csv")
        df = pd.read_csv(file_path)
        col = df.values.flatten()
        labels = col.tolist()
        print("Sending labels:", labels)
        return jsonify({'labels': labels})
    except Exception as e:
        print(f"Error in get_labels: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    try:
        print("Received request for models")
        models = [
            {"label": "Trained Model", "value": "trained_model"},
            {"label": "Megatron Model_e100", "value": "megatron_e100"},
            {"label": "Alpha Tuned Model_e1k", "value": "alpha_tuned_model_e1k_"},
            {"label": "GOAT Model_e10k", "value": "GOAT_model_e1k_"},
            {"label": "Natural Model_e10", "value": "natural_model_e10_"},
            {"label": "Pushed Model_e50", "value": "pushed_model_e50_"},
            {"label": "Massive Surya_e10k", "value": "massive_surya_e10k"}
        ]
        print("Sending models:", models)
        return jsonify({'models': models})
    except Exception as e:
        print(f"Error in get_models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received prediction request with data:", data)
        model_path = os.path.join(BASE_DIR, "ML", "Model", f"{data['model']}.pth")
        adj_matrix = np.loadtxt(os.path.join(BASE_DIR, "ML", "Data", "adj.csv"), delimiter=",")
        wgt_matrix = np.loadtxt(os.path.join(BASE_DIR, "ML", "Data", "label.csv"), delimiter=",")
        df = pd.read_csv(os.path.join(BASE_DIR, "ML", "Data", "labelele.csv"))
        col = df.values.flatten()
        node_labels = {i: label for i, label in enumerate(col)}
        model = cs.GraphTransformer(num_layers=2, d_model=128, num_heads=8, d_feedforward=512, input_dim=len(adj_matrix[0]), use_weights=True)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        node_index = list(node_labels.keys())[list(node_labels.values()).index(data['label'])]
        recommended_indices = cs.predict(model, adj_matrix, node_index, top_k=data['topK'], threshold=data['threshold'])
        recommended_labels = [node_labels.get(idx, "Label not found") for idx in recommended_indices]
        print("Sending recommendations:", recommended_labels)
        return jsonify({'recommendations': recommended_labels})
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)