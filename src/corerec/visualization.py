from common_import import *


def draw_graph(adj_matrix, top_nodes, recommended_nodes=None):
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)

    node_colors = []
    for node in G.nodes():
        if recommended_nodes is not None and node in recommended_nodes:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)

    if top_nodes is not None:
        top_node_color = 'green'
        nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_color=top_node_color, node_size=500, node_shape='s')

    plt.title("Recommended Nodes Highlighted in Blue and Top Nodes in Red")
    plt.show()


