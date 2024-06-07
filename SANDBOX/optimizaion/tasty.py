# import numpy as np
# import TCR as cr
# import vish_graphs as vg
# from common_import import *
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import networkx as nx
# from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor

# def format_predictions(predictions):
#     formatted_results = []
#     for pred in predictions:
#         formatted_results.append(
#             f"Node: {pred['node']}\n"
#             f"Score: {pred['score']:.4f}\n"
#             f"Jaccard Similarity: {pred['jaccard_similarity']:.4f}\n"
#             f"Adamic/Adar Index: {pred['adamic_adar_index']:.4f}\n"
#             f"Explanation: {pred['explanation']}\n"
#             "-----------------------------"
#         )
#     return "\n".join(formatted_results)

# if __name__ == '__main__':
#     adj_matrix = np.random.rand(100, 100)  # Example larger adjacency matrix
#     weight_matrix = np.random.rand(100, 100)  # Example larger weight matrix

#     dataset = cr.GraphDataset(adj_matrix, weight_matrix)
#     data_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)  # Use more workers for larger data

#     model = cr.GraphTransformer(num_layers=4, d_model=128, num_heads=8, d_feedforward=256, input_dim=100, use_weights=True)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     cr.train_model(model, data_loader, criterion, optimizer, num_epochs=2)

#     # Predict and explain
#     node_index = 0
#     recommended_indices, explanations = cr.explainable_predict(model, adj_matrix, node_index)
#     print("Recommended Indices:", recommended_indices)
#     print("Explanations:")
#     print(cr.format_predictions(explanations))

#     # Example of parallel processing for similarity scores
#     def compute_similarity_scores(node):
#         return cr.similarity_scores(adj_matrix, node, 'jaccard')

#     nodes_to_compute = list(range(100))  # Example nodes
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         results = list(executor.map(compute_similarity_scores, nodes_to_compute))

#     print("Similarity Scores for Nodes:", results)
import re

def convert_function_definition(line):
    match = re.match(r'def (\w+)\((.*)\):', line)
    if match:
        function_name = match.group(1)
        arguments = match.group(2)
        return_type = 'auto'  # Default return type
        if function_name == 'fibonacci':
            return_type = 'std::vector<int>'  # Adjust return type for Fibonacci function
        return f'{return_type} {function_name}({arguments}) {{'
    return None

def convert_while_loop(line):
    match = re.match(r'while (.+):', line)
    if match:
        condition = match.group(1)
        return f'while ({condition}) {{'
    return None

def convert_append_statement(line):
    match = re.match(r'(\w+)\.append\((.+)\)', line)
    if match:
        list_name = match.group(1)
        value = match.group(2)
        return f'{list_name}.push_back({value});'
    return None

def convert_return_statement(line):
    if line.strip().startswith('return '):
        return line.replace('return ', 'return ')
    return None

def convert_python_to_cpp(python_code):
    cpp_code = []
    function_definitions = []

    # Split the code into lines
    lines = python_code.split('\n')

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            continue

        converted_line = convert_line(stripped_line)
        if converted_line:
            cpp_code.append(converted_line)

        # Check if the line is a function definition
        if stripped_line.startswith('def '):
            function_definitions.append(convert_function_definition(stripped_line))

    # Add includes
    cpp_code.insert(0, '#include <iostream>')
    cpp_code.insert(1, '#include <string>')
    cpp_code.insert(2, '#include <algorithm>')
    cpp_code.insert(3, '#include <vector>')
    cpp_code.append('\n')

    # Add function definitions
    cpp_code.extend(function_definitions)

    # Add int main() function
    cpp_code.append('int main() {')
    cpp_code.append('    // Call your function with example use cases here')
    cpp_code.append('    return 0;')
    cpp_code.append('}')

    cpp_code_str = '\n'.join(cpp_code)

    return cpp_code_str

def convert_line(line):
    converters = [
        convert_while_loop,
        convert_append_statement,
        convert_return_statement
    ]
    
    for converter in converters:
        converted_line = converter(line)
        if converted_line:
            return converted_line

    return f'// {line} (unrecognized line)'

# Test code
python_code = """
def fibonacci(n):
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib

n = 10  # Change this to generate Fibonacci sequence up to a different number
print(fibonacci(n))
"""

cpp_code = convert_python_to_cpp(python_code)
print(cpp_code)
