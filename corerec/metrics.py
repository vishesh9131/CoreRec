"""
metrics.py

This module provides functions to calculate various graph-based metrics.

Functions:
    jaccard_similarity(graph, node): Calculates the Jaccard similarity for a given node.
    adamic_adar_index(graph, node): Calculates the Adamic-Adar index for a given node.
    aaj_accuracy(graph, node_index, recommended_indices): Calculates the average Jaccard and Adamic-Adar indices for recommended nodes.

Usage:
    from engine.metrics import jaccard_similarity, adamic_adar_index, aaj_accuracy

    # Example usage
    graph = np.array([[0, 1], [1, 0]])
    node = 0
    print(jaccard_similarity(graph, node))
    print(adamic_adar_index(graph, node))
    print(aaj_accuracy(graph, node, [1]))
"""

import networkx as nx
import numpy as np

def jaccard_similarity(graph, node):
    G = nx.from_numpy_array(graph)
    scores = []
    neighbors = set(G.neighbors(node))
    for n in G.nodes():
        if n != node:
            neighbors_n = set(G.neighbors(n))
            intersection = len(neighbors & neighbors_n)
            union = len(neighbors | neighbors_n)
            score = intersection / union if union != 0 else 0
            scores.append((n, score))
    return scores

def adamic_adar_index(graph, node):
    G = nx.from_numpy_array(graph)
    scores = []
    neighbors = set(G.neighbors(node))
    for n in G.nodes():
        if n != node:
            neighbors_n = set(G.neighbors(n))
            shared_neighbors = neighbors & neighbors_n
            score = sum(1 / np.log(len(list(G.neighbors(nn)))) for nn in shared_neighbors if len(list(G.neighbors(nn))) > 1)
            scores.append((n, score))
    return scores

def aaj_accuracy(graph, node_index, recommended_indices):
    G = nx.from_numpy_array(graph)
    jaccard_scores = []
    adamic_adar_scores = []

    for rec_node in recommended_indices:
        preds = list(nx.jaccard_coefficient(G, [(node_index, rec_node)]))
        if preds:
            jaccard_scores.append(preds[0][2])

        preds = list(nx.adamic_adar_index(G, [(node_index, rec_node)]))
        if preds:
            adamic_adar_scores.append(preds[0][2])

    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
    avg_adamic_adar = np.mean(adamic_adar_scores) if adamic_adar_scores else 0

    return avg_jaccard, avg_adamic_adar