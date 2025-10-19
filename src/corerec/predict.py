"""
predict.py

This module provides functions to predict top-k nodes based on model outputs, with optional thresholding.

Functions:
    predict(model, graph, node_index, top_k=5, threshold=0.5): Predicts top-k nodes based on model outputs.
    explainable_predict(model, graph, node_index, top_k=5, threshold=0.5): Predicts top-k nodes with explanations based on graph metrics.

Usage:
    from engine.predict import predict, explainable_predict

    # Example usage
    model = GraphTransformer(num_layers=2, d_model=128, num_heads=4, d_feedforward=512, input_dim=10)
    graph = np.array([[0, 1], [1, 0]])
    node_index = 0
    print(predict(model, graph, node_index))
    print(explainable_predict(model, graph, node_index))
"""

from corerec.common_import import *

def predict(model, graph, node_index, top_k=5, threshold=0.5):
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(graph[node_index]).unsqueeze(0)
        output = model(input_data)
        scores = output.squeeze().numpy()
    
    recommended_indices = [i for i, score in enumerate(scores) if score > threshold]
    recommended_indices = sorted(recommended_indices, key=lambda i: scores[i], reverse=True)[:top_k]
    return recommended_indices

def explainable_predict(model, graph, node_index, top_k=5, threshold=0.5):
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(graph[node_index]).unsqueeze(0)
        output = model(input_data)
        scores = output.squeeze().numpy()

    recommended_indices = [i for i, score in enumerate(scores) if score > threshold]
    recommended_indices = sorted(recommended_indices, key=lambda i: scores[i], reverse=True)[:top_k]

    explanations = []
    G = nx.from_numpy_array(graph)
    user_neighbors = set(G.neighbors(node_index))

    for idx in recommended_indices:
        node_neighbors = set(G.neighbors(idx))
        jaccard = len(user_neighbors & node_neighbors) / len(user_neighbors | node_neighbors) if len(user_neighbors | node_neighbors) > 0 else 0
        adamic_adar = sum(1 / np.log(len(list(G.neighbors(nn)))) for nn in user_neighbors & node_neighbors if len(list(G.neighbors(nn))) > 1)

        explanation_text = f"The recommendation is based on the similarity of node {idx} to your interests and its connections to relevant nodes."

        explanation = {
            "node": idx,
            "score": scores[idx],
            "jaccard_similarity": jaccard,
            "adamic_adar_index": adamic_adar,
            "explanation": explanation_text
        }
        explanations.append(explanation)

    return recommended_indices, explanations