cdef extern from "vish_graphs.c":
    void generate_random_graph(int num_people, const char *file_path, unsigned int seed)
    void generate_weight_matrix(int num_nodes, int weight_min, int weight_max, const char *file_path, unsigned int seed)

def py_generate_random_graph(int num_people, str file_path, int seed):
    generate_random_graph(num_people, file_path.encode('utf-8'), seed)

def py_generate_weight_matrix(int num_nodes, int weight_min, int weight_max, str file_path, int seed):
    generate_weight_matrix(num_nodes, weight_min, weight_max, file_path.encode('utf-8'), seed)