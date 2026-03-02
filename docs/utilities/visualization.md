# VishGraphs Visualization

**VishGraphs** is the built-in visualization engine of CoreRec. It allows you to visualize 2D and 3D embeddings of your user-item graphs, helping you understand the structure of your data.

## 2D Visualization

Use `draw_graph` to plot a standard 2D network.

```python
import corerec.vish_graphs as vg

# 1. Generate or Load Adjacency Matrix
adj_matrix = vg.generate_random_graph(num_people=50)

# 2. Draw
vg.draw_graph(
    adj_matrix, 
    top_nodes=[0, 1, 2],  # Highlight specific nodes
    node_labels=["User A", "User B", ...]
)
```

## 3D Visualization

For complex networks, 3D visualization can reveal clusters more effectively.

```python
vg.draw_graph_3d(
    adj_matrix,
    top_nodes=[0, 1, 2],
    transparent_labeled=True
)
```

## Bipartite Relationships

Visualize the connection between two distinct sets of nodes (e.g., Users and Items).

```python
vg.show_bipartite_relationship(adj_matrix)
```

## Gallery

### Random Graph Generation
![Random Graph](../images/intro.png)

### 3D Network Structure
*(Placeholder for 3D interactive graph)*

!!! note "Interactive Plots"
    VishGraphs uses `matplotlib` for static plots but can be extended to `plotly` for interactive web-based visualizations in the future.
