# Graph Based Algorithms

Graph-based methods model data as a user-item bipartite graph. They are powerful for capturing high-order connectivity (e.g., "users who bought what I bought also bought X").

## Overview

These models propagate embeddings through the graph structure to smooth signals and alleviate sparsity.

## Available Models

### LightGCN
A simplified GCN that removes non-linearities and feature transformations, making it highly efficient for recommendation.

::: corerec.engines.collaborative.graph_based_base.lightgcn.LightGCN
    options:
      show_root_heading: true
      show_source: true

### NGCF (Neural Graph Collaborative Filtering)
Explicitly models the high-order connectivity in the user-item graph.

::: corerec.engines.collaborative.graph_based_base.ngcf.NGCF
    options:
      show_root_heading: true
      show_source: true
