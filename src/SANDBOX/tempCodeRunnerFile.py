import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Load node labels from the CSV file
df_labels = pd.read_csv("SANDBOX/labelele.csv")
node_labels = df_labels["Names"].tolist()

# Assuming you have the recommended nodes for each node index stored in a list of lists
recommended_nodes_list = []
for i in range(11):
    recommended_nodes = cs.predict(model, adj_matrix, i, top_k=3, threshold=0.7)
    recommended_nodes_list.append(recommended_nodes)

# Create a scatter plot with labels for node indexes
plt.figure(figsize=(10, 8))
for i in range(11):
    x = [i] * len(recommended_nodes_list[i])
    y = recommended_nodes_list[i]
    plt.scatter(x, y, label=f"Node {i}")
    for j, txt in enumerate(y):
        plt.annotate(node_labels[y[j]], (x[j], y[j]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xticks(range(11), node_labels, rotation=45)
plt.yticks(range(11), node_labels)
plt.xlabel('Node Index / Names')
plt.ylabel('Recommended Node Index / Names')
plt.title('Recommendations for Node Index')

# Create a legend table with batch size and threshold details outside the plot area
legend_data = [['Batch Size', 'Threshold'],
               [10, 0.7]]
table = plt.table(cellText=legend_data, loc='upper right', cellLoc='center', colWidths=[0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 1.5)

plt.grid(True)
plt.show()