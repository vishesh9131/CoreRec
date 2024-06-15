import ctypes
import os
import time
# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "vish_graphs.so")
lib = ctypes.CDLL(lib_path)

# Define the argument and return types for the C functions
lib.generate_random_graph.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
lib.generate_weight_matrix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]

def generate_random_graph(num_people, file_path="graph_dataset.csv", seed=None):
    if seed is None:
        seed = int(time.time())
    lib.generate_random_graph(num_people, file_path.encode('utf-8'), seed)

def generate_weight_matrix(num_nodes, weight_range=(1, 10), file_path="weight_matrix.csv", seed=None):
    if seed is None:
        seed = int(time.time())
    lib.generate_weight_matrix(num_nodes, weight_range[0], weight_range[1], file_path.encode('utf-8'), seed)

# Example usage
generate_random_graph(100, "786n.csv", 42)
generate_weight_matrix(100, (1, 10), "786w.csv", 42)

