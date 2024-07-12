import torch
import numpy as np
import networkx as nx


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