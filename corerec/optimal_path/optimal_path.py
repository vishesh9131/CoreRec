import itertools
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def tsp(graph, start):
    n = len(graph)
    all_visited = (1 << n) - 1
    memo = {}
    path_memo = {}

    def visit(city, visited):
        if visited == all_visited:
            return graph[city][start], [start]
        if (city, visited) in memo:
            return memo[(city, visited)], path_memo[(city, visited)]

        min_cost = float('inf')
        min_path = []
        for next_city in range(n):
            if visited & (1 << next_city) == 0:
                cost, path = visit(next_city, visited | (1 << next_city))
                cost += graph[city][next_city]
                if cost < min_cost:
                    min_cost = cost
                    min_path = [next_city] + path

        memo[(city, visited)] = min_cost
        path_memo[(city, visited)] = min_path
        return min_cost, min_path

    min_cost, path = visit(start, 1 << start)
    path = [start] + path
    return min_cost, path, path_memo

def show_path(graph, start_city):
    min_cost, path, path_memo = tsp(graph, start_city)
    print(f"Minimum path cost: {min_cost}")
    print(f"Path: {path}")

    fig, ax = plt.subplots()

    G = nx.DiGraph()
    n = len(graph)
    for (city, visited), next_cities in path_memo.items():
        for next_city in next_cities:
            G.add_edge((city, visited), (next_city, visited | (1 << next_city)))

    # Add subset attribute to each node
    for node in G.nodes:
        G.nodes[node]['subset'] = bin(node[1]).count('1')

    # Position nodes in a tree layout
    pos = nx.multipartite_layout(G, subset_key='subset')

    # Draw the tree
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

    def update(frame):
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

        if frame < len(path) - 1:
            edges = []
            visited = 1 << path[0]
            for i in range(frame + 1):
                edge = [(path[i], visited), (path[i+1], visited | (1 << path[i+1]))]
                if edge[0] in pos and edge[1] in pos:  # Ensure both nodes are in pos
                    edges.append(edge)
                visited |= (1 << path[i+1])
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2.5, ax=ax)

    ani = animation.FuncAnimation(fig, update, frames=len(path)-1, interval=1000, repeat=False)

    plt.show()