<div align="center">
    <img src="docs/images/coreRec.svg" style="vertical-align: middle; margin-right: 0px; margin-bottom: 20px;" width="70" height="70">
    <h1>CoreRec & VishGraphs Manual</h1>
</div>

Discover the power of graph analysis and recommendation with CoreRec & VishGraphs. Dive into our comprehensive manual and explore the endless possibilities.

<h2>
    <img src="docs/images/intro.png" style="vertical-align: middle; margin-right: 8px;" width="30" height="30">
    Introduction
</h2>

CoreRec is your all-in-one recommendation engine for graph-based algorithms. Seamlessly integrating advanced neural network architectures, CoreRec excels in node recommendations, model training, and graph visualizations, making it the ultimate tool for data scientists and researchers.
VishGraphs is your ultimate Python library for graph visualization and analysis. Whether you're a data scientist, researcher, or hobbyist, VishGraphs offers intuitive tools to generate, visualize, and analyze graphs effortlessly.

<h2>
    <img src="docs/images/feature.png" style="vertical-align: middle; margin-right: 10px;" width="40" height="38">
    Features
</h2>


#### core_rec.py

- **`GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)`**
  - A Transformer model for graph data with customizable parameters.
  
- **`GraphDataset(adj_matrix)`**
  - A PyTorch dataset for graph data, streamlining model training.
  
- **`train_model(model, data_loader, criterion=False, optimizer=False, num_epochs=False)`**
  - Train your model with ease using our flexible training function.
  
- **`predict(model, graph, node_index, top_k=5)`**
  - Predict similar nodes with precision using trained models.
  
- **`aaj_accuracy(graph, node_index, recommended_indices)`**
  - Measure the accuracy of your recommendations with our robust metrics.

#### vish_graphs.py

- **`generate_large_random_graph(num_people, file_path="large_random_graph.csv", seed=None)`**
  - Generate and save large random graphs effortlessly.
  
- **`draw_graph(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None, transparent_labeled=True, edge_weights=None)`**
  - Create stunning 2D visualizations of your graphs.
  
- **`draw_graph_3d(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None, transparent_labeled=True, edge_weights=None)`**
  - Experience your graphs in 3D with customizable features.
  
- **`show_bipartite_relationship(adj_matrix)`**
  - Visualize bipartite relationships with clarity.

<h2>
    <img src="docs/images/install.png" style="vertical-align: middle; margin-right: 10px;" width="40" height="38">
    Installation
</h2>

For security reasons, we are not providing the pip install command at this time.

Simply copy the files "`vish_graphs.py`", "`core_rec.py`", and "`common_imports.py`" to your project folder and import them.

### Or;

### For Windows Users  
Set the Python path using the `set` command or copy and paste this in the command prompt:

```
set PATH "%PATH%;C:\path\to\your\CoreRecRepo"
```

### For Mac Users

Set the Python path using the `export` command or copy and paste this in the terminal:

```
export PATH="$PATH:/path/to/your/CoreRecRepo"
```


<h2>
    <img src="docs/images/coreRec.svg" style="vertical-align: middle; margin-right: 0px;" width="40" height="40">
    CoreRec Manual
</h2>

<h3>
    <img src="docs/images/star.png" style="vertical-align: middle; margin-right: 0px;" width="20" height="20">
    Table of Contents
</h3>

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Generating Random Graphs](#generating-random-graphs)
    - [Drawing Graphs](#drawing-graphs)
4. [Directory Structure](#directory-structure)
5. [Troubleshooting](#troubleshooting)
6. [Contributing](#contributing)
7. [License](#license)

# Introduction
- CoreRec:
CoreRec is a cutting-edge recommendation engine for graph data analysis and visualization. It excels in recommending similar nodes, training machine learning models, and visualizing complex network structures.
- VishGraphs:
VishGraphs is a Python library designed for graph visualization and analysis. It offers tools for generating random graphs, drawing graphs in 2D and 3D, and analyzing graph properties.

<h3>
    <img src="docs/images/struct.png" style="vertical-align: middle; margin-right: 0px;" width="40" height="40">
    Directory Structure
</h3>

<table>
<thead>
<tr>
<th> Description </th>
<th> Quick Access </th>
</tr>
</thead>
<tbody>
<!-- Row boilerplate (copy-paste the following commented snippet for adding a new row to the table.)
<tr> <td> <h3> title </h3> 
description
</td> <td> <pre>
folders
</pre> </td> </tr>
-->
<tr> <td> <h3> engine </h3> 
Contains core engine components and utilities.
</td> <td> <pre>
└── engine
    ├── cr_boosters/
    ├── cr_pkg/
    ├── cr_utility/
    ├── torch_nn/
    ├── async_dpp.py
    ├── common_import.py
    ├── core_rec.py
    ├── datasets.py
    ├── metrics.py
    ├── model.py
    ├── predict.py
    ├── timecapsule.py
    ├── Tmodel.py
    ├── train.py
    ├── vish_graphs.py
    └── visulization.py
</pre> </td> </tr>

<tr> <td> <h3> Roadmap </h3> 
Contains documents related to future updates and use cases.
</td> <td> <pre>
└── roadmap
    ├── futureupdates.md
    ├── Readme.pdf
    └── usecases.pdf
</pre> </td> </tr>

<tr> <td> <h3> src </h3> 
Contains the main source code and use cases.
</td> <td> <pre>
└── src
    ├── backup/
    ├── CoreRec/
    ├── SANDBOX/
    └── USECASES
        ├── custommodel.py
        ├── customthreshold.py
        ├── labels.py
        └── custommodel.py
</pre> </td> </tr>

<tr> <td> <h3> Vish Graphs </h3> 
Contains build and distribution files for Vish Graphs.
</td> <td> <pre>
└── vish_graphs
    ├── build/
    ├── dist/
    └── vish_graphs/
</pre> </td> </tr>

</tbody>
</table>


# Usage
### Generating Random Graphs
Generate random graphs effortlessly with the `generate_random_graph` function:

```python
import vish_graphs as vg
graph_file = vg.generate_random_graph(10, "random_graph.csv")
```

# The use cases are:-
## 🔍 Delve into Advanced Graph Analysis and Recommendation with VishGraphs and CoreRec! 🚀
Welcome to a world of cutting-edge graph analysis and recommendation tools brought to you by VishGraphs and CoreRec. Uncover the potential of data visualization and machine learning in a sophisticated manner.

[🔗 Explore Detailed UseCases Here 🔗](https://github.com/vishesh9131/CoreRec/blob/main/USECASES/usecases.md)

## CoreRec

```python
import core_rec as cs
```
### 1. `recommend_similar_nodes(adj_matrix, node)`

Recommends similar nodes based on cosine similarity scores calculated from the adjacency matrix.

**Use case:** Providing recommendations for nodes based on their similarity within a graph.

### 2. `GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)`

Defines a Transformer model for graph data with customizable parameters.

**Use case:** Training machine learning models for graph-related tasks, such as node classification or link prediction.

### 3. `GraphDataset(adj_matrix)`

Defines a PyTorch dataset for graph data, allowing easy integration with DataLoader for model training.

**Use case:** Preparing graph data for training machine learning models.

### 4. `train_model(model, data_loader, criterion, optimizer, num_epochs)`

Trains a given model using the provided data loader, loss function, optimizer, and number of epochs.

**Use case:** Training machine learning models for graph-related tasks using graph data.

In the `test.py` file, various functionalities from `vishgraphs.py` and `core_rec.py` are utilized and demonstrated:
- Random graph generation (`generate_random_graph`).
- Identification of top nodes in a graph (`find_top_nodes`).
- Training a Transformer model for graph data (`GraphTransformer`, `GraphDataset`, `train_model`).
- Recommending similar nodes using a trained model (`recommend_similar_nodes`).
- Visualization of a graph in 3D (`draw_graph_3d`).

## vishgraphs

```python
import vishgraphs as vg
```
### 1. `generate_random_graph(num_people, file_path="graph_dataset.csv", seed=None)`

Generate a random graph with a specified number of people and save the adjacency matrix to a CSV file.

**Use case:** Generating synthetic graph data for testing algorithms or simulations.

### 2. `draw_graph(adj_matrix, nodes, top_nodes)`

Draw a 2D visualization of a graph based on its adjacency matrix, highlighting top nodes if specified.

**Use case:** Visualizing relationships within a graph dataset.

### 3. `find_top_nodes(matrix, num_nodes=10)`

Identify the top nodes with the greatest number of strong correlations in a graph.

**Use case:** Identifying influential or highly connected nodes in a network.

### 4. `draw_graph_3d(adj_matrix, nodes, top_nodes)`

Create a 3D visualization of a graph based on its adjacency matrix, with optional highlighting of top nodes.

**Use case:** Visualizing complex network structures in a three-dimensional space.

### 5. `show_bipartite_relationship_with_cosine(adj_matrix)`

Visualize bipartite relationships in a graph using cosine similarity and community detection algorithms.

**Use case:** Analyzing relationships between different sets of nodes in a bipartite graph.

### 6. `bipartite_matrix_maker(csv_path)`

Read a CSV file containing a bipartite adjacency matrix and return it as a list.

**Use case:** Preparing data for analyzing bipartite networks.

---

Explore the codebase and utilize these functionalities for your graph analysis and recommendation tasks! If you have any questions or need further assistance, don't hesitate to reach out. Happy graph analyzing! 📊🔍

### Drawing Graphs
VishGraphs supports drawing graphs in both 2D and 3D:

```python
adj_matrix = vishgraphs.bipartite_matrix_maker(graph_file)
nodes = list(range(len(adj_matrix)))
top_nodes = [0, 1, 2] # Example top nodes
vishgraphs.draw_graph(adj_matrix, nodes, top_nodes)
```

<h3>
    <img src="docs/images/trouble.png" style="vertical-align: middle; margin-right: 0px;" width="40" height="40">
    Troubleshooting
</h3>

### Troubleshooting Guide

For issues with CoreRec and VishGraphs:

1. **Check Documentation:** Ensure you're following the library's guidelines and examples correctly.
2. **GitHub Issues:** Report bugs or seek help by creating an issue on the GitHub repository.
3. **Verify Data:** Confirm that your input data is correctly formatted and compatible.
4. **Model Parameters:** Double-check model configurations and training parameters.
5. **Visualization Inputs:** Ensure correct parameters for graph visualization functions.
6. **Community Help:** Utilize community forums for additional support.

This streamlined approach should help resolve common issues efficiently.

<h3>
    <img src="docs/images/cont.png" style="vertical-align: middle; margin-right: 0px;" width="40" height="40">
    Contributing
</h3>

We welcome contributions to enhance the functionalities of our graph analysis and recommendation tools. If you're interested in contributing, here are a few ways you can help:

- **Bug Fixes:** Identify and fix bugs in the existing code.
- **Feature Enhancements:** Suggest and implement improvements to current features.
- **New Features:** Propose and develop new features that could benefit users of the libraries.
- **Documentation:** Help improve the documentation to make the libraries more user-friendly.

### To contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Develop your changes while adhering to the coding standards and guidelines.
4. Submit a pull request with a clear description of the changes and any relevant issue numbers.

Your contributions are greatly appreciated and will help make these tools more effective and accessible to everyone!

<h3>
    <img src="docs/images/lic.png" style="vertical-align: middle; margin-right: 0px;" width="40" height="40">
    License
</h3>
VishGraphs is distributed under the following terms:

>The library and utilities are only for research purposes. Please do not use it commercially without the author's (@Vishesh9131) consent.
