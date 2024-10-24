# Benchmarking CoreRec with Various Graph-Based Models

This document provides a summary of the models used to benchmark the `CoreRec` model. The models are categorized into different types of graph-based algorithms.

| Category                          | Model Name                        | Description                                                                                       |
|-----------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------|
| **Graph Neural Networks (GNNs)**  | Graph Convolutional Networks (GCNs) | Extend convolutional networks to graph data, capturing local neighborhood information.            |
|                                   | Graph Attention Networks (GATs)    | Use attention mechanisms to weigh the importance of neighboring nodes differently.                |
|                                   | GraphSAGE                          | Generates node embeddings in an inductive manner, handling new, unseen nodes during training.     |
| **Knowledge Graph Embedding (KGE)** | TransE                            | Embeds entities and relationships of knowledge graphs into continuous vector spaces.              |
|                                   | TransR                             | Similar to TransE but considers different types of relationships.                                 |
|                                   | DistMult                           | Factorizes the adjacency matrix of the knowledge graph.                                           |
|                                   | ComplEx                            | Extends DistMult by using complex-valued embeddings.                                              |
| **Heterogeneous Graph Embedding** | Heterogeneous Graph Attention Networks (HAN) | Handles graphs with different types of nodes and edges, capturing rich information.               |
|                                   | MetaPath2Vec                       | Generates embeddings for heterogeneous networks by leveraging metapaths.                          |
| **Collaborative Filtering on Graphs** | Graph-based Collaborative Filtering (GCF) | Incorporates graph structures into traditional collaborative filtering methods.                   |
|                                   | Graph Regularized Matrix Factorization (GRMF) | Integrates graph regularization into matrix factorization.                                        |
| **Graph-based Sequential Recommendation** | STAGE (Self-Attentive Graph Embedding) | Combines graph embedding with self-attention to capture sequential patterns in user behavior.     |
|                                   | SR-GNN (Session-based Recommendation with GNN) | Uses GNNs to model user sessions as graphs.                                                       |
| **Random Walk Based Algorithms**  | DeepWalk                           | Generates node embeddings by performing random walks on the graph.                                |
|                                   | Node2Vec                           | An extension of DeepWalk, adding flexibility in exploring neighborhoods.                          |
| **Meta Exploit**                  | Meta Exploit                       | The model being benchmarked against the above models.                                             |

## Usage

To benchmark the `CoreRec` model against the above models, follow the steps in the provided Jupyter notebook.

## Results

The results of the benchmarking will be visualized in a bar chart, comparing various metrics such as precision, recall, F1 score, accuracy, specificity, and sensitivity across all models.



# Benchmark Scales Defination
| Metric       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| Precision    | The ratio of correctly predicted positive observations to the total predicted positives. |
| Recall       | The ratio of correctly predicted positive observations to all observations in the actual class. |
| F1 Score     | The weighted average of Precision and Recall. It considers both false positives and false negatives. |
| Accuracy     | The ratio of correctly predicted observations to the total observations. |
| Specificity  | The ratio of correctly predicted negative observations to all observations in the actual negative class. |
| Sensitivity  | Another term for Recall. |
| ROC AUC      | The area under the receiver operating characteristic curve. It measures the ability of the model to distinguish between classes. |
| MCC          | Matthews correlation coefficient. It takes into account true and false positives and negatives and is generally regarded as a balanced measure. |


# Benchmark Results

This table summarizes the benchmark results for various models, including `CoreRec`, across different metrics.

| model        |   precision |   recall |   f1_score |   accuracy |   specificity |   sensitivity |   roc_auc |      mcc |
|:-------------|------------:|---------:|-----------:|-----------:|--------------:|--------------:|----------:|---------:|
| CoreRec      |    0.621212 | 0.727273 |   0.651515 |   0.727273 |      0.621212 |      0.975207 |      0.85 | 0.726933 |
| GCN          |    0.149351 | 0.272727 |   0.174242 |   0.272727 |      0.149351 |      0.933058 |      0.6  | 0.258199 |
| GraphSAGE    |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |
| TransE       |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |
| TransR       |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |
| DistMult     |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |
| ComplEx      |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |
| HAN          |    0.621212 | 0.727273 |   0.651515 |   0.727273 |      0.621212 |      0.975207 |      0.85 | 0.726933 |
| MetaPath2Vec |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |
| GCF          |    0.621212 | 0.727273 |   0.651515 |   0.727273 |      0.621212 |      0.975207 |      0.85 | 0.726933 |
| GRMF         |    0.621212 | 0.727273 |   0.651515 |   0.727273 |      0.621212 |      0.975207 |      0.85 | 0.726933 |
| GAT          |    0.2      | 0.272727 |   0.212121 |   0.272727 |      0.2      |      0.929162 |      0.6  | 0.237508 |
| STAGE        |    0.621212 | 0.727273 |   0.651515 |   0.727273 |      0.621212 |      0.975207 |      0.85 | 0.726933 |
| SR-GNN       |    0.621212 | 0.727273 |   0.651515 |   0.727273 |      0.621212 |      0.975207 |      0.85 | 0.726933 |
| DeepWalk     |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |
| Node2Vec     |    0.590909 | 0.727273 |   0.636364 |   0.727273 |      0.590909 |      0.975207 |      0.85 | 0.719909 |






|   models     |   jaccard |
|:-------------|----------:|
| CoreRec      |  0.621212 |
| GCN          |  0.149351 |
| GraphSAGE    |  0.590909 |
| TransE       |  0.621212 |
| TransR       |  0.590909 |
| DistMult     |  0.621212 |
| ComplEx      |  0.590909 |
| HAN          |  0.590909 |
| MetaPath2Vec |  0.621212 |
| GCF          |  0.636364 |
| GRMF         |  0.590909 |
| GAT          |  0.222727 |
| STAGE        |  0.621212 |
| SR-GNN       |  0.590909 |
| DeepWalk     |  0.590909 |
| Node2Vec     |  0.590909 |






# Models Performance in Dataset of 500 nodes 

| Models        | Precision       | Recall          | F1 Score        | Accuracy        | Specificity     | Sensitivity     | ROC AUC         | MCC             |
|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| CoreRec       | 0.00323017      | **0.016**       | 0.00420422      | **0.016**       | 0.00323017      | **0.998032**    | **0.507014**    | 0.0268423       |
| GCN           | 4e-06           | 0.002           | 7.98403e-06     | 0.002           | 4e-06           | 0.996004        | 0.5             | 0               |
| GraphSAGE     | **0.0140041**   | **0.016**       | **0.0140081**   | **0.016**       | **0.0140041**   | **0.998032**    | **0.507014**    | **0.0840868**   |
| TransE        | 0.00026408      | **0.016**       | 0.000519333     | **0.016**       | 0.00026408      | **0.998032**    | **0.507014**    | 0.0150155       |
| TransR        | 0.000313148     | **0.016**       | 0.000611743     | **0.016**       | 0.000313148     | **0.998032**    | **0.507014**    | 0.0151607       |
| DistMult      | 0.000315137     | **0.016**       | 0.000615142     | **0.016**       | 0.000315137     | **0.998032**    | **0.507014**    | 0.015194        |
| ComplEx       | 0.000292928     | **0.016**       | 0.000573907     | **0.016**       | 0.000292928     | **0.998032**    | **0.507014**    | 0.0151355       |
| HAN           | 0.000262374     | **0.016**       | 0.000516088     | **0.016**       | 0.000262374     | **0.998032**    | **0.507014**    | 0.0150088       |
| MetaPath2Vec  | 0.00029189      | **0.016**       | 0.000571624     | **0.016**       | 0.00029189      | **0.998032**    | **0.507014**    | 0.0150935       |
| GCF           | 0.000266522     | **0.016**       | 0.000523983     | **0.016**       | 0.000266522     | **0.998032**    | **0.507014**    | 0.0150004       |
| GRMF          | 0.000274133     | **0.016**       | 0.000538482     | **0.016**       | 0.000274133     | **0.998032**    | **0.507014**    | 0.0150655       |
| GAT           | 4e-06           | 0.002           | 7.98403e-06     | 0.002           | 4e-06           | 0.996004        | 0.5             | 0               |
| STAGE         | 0.000268031     | **0.016**       | 0.000526806     | **0.016**       | 0.000268031     | **0.998032**    | **0.507014**    | 0.0150272       |
| SR-GNN        | 0.000272428     | **0.016**       | 0.000535193     | **0.016**       | 0.000272428     | **0.998032**    | **0.507014**    | 0.0150485       |
| DeepWalk      | 0.00025766      | **0.016**       | 0.000507101     | **0.016**       | 0.00025766      | **0.998032**    | **0.507014**    | 0.0149883       |
| Node2Vec      | 0.000281149     | **0.016**       | 0.000551475     | **0.016**       | 0.000281149     | **0.998032**    | **0.507014**    | 0.0150604       |