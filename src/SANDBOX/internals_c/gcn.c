#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to perform matrix multiplication
void matmul(float *a, float *b, float *c, int m, int n, int p) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j) {
            c[i * p + j] = 0;
            for (int k = 0; k < n; ++k)
                c[i * p + j] += a[i * n + k] * b[k * p + j];
        }
}

// Function to normalize the adjacency matrix
void normalize_adj(float *adj, float *norm_adj, int num_nodes) {
    float *degree = (float *)malloc(num_nodes * sizeof(float));
    
    // Calculate degree matrix
    for (int i = 0; i < num_nodes; ++i) {
        degree[i] = 0;
        for (int j = 0; j < num_nodes; ++j)
            degree[i] += adj[i * num_nodes + j];
    }
    
    // Normalize adjacency matrix
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (degree[i] > 0 && degree[j] > 0)
                norm_adj[i * num_nodes + j] = adj[i * num_nodes + j] / sqrt(degree[i] * degree[j]);
            else
                norm_adj[i * num_nodes + j] = 0;
        }
    }
    
    free(degree);
}

// GCN layer forward pass
void gcn_layer(float *x, float *adj, float *w, float *out, int num_nodes, int input_dim, int output_dim) {
    float *norm_adj = (float *)malloc(num_nodes * num_nodes * sizeof(float));
    float *temp = (float *)malloc(num_nodes * input_dim * sizeof(float));
    
    // Normalize adjacency matrix
    normalize_adj(adj, norm_adj, num_nodes);
    
    // Matrix multiplication: XW
    matmul(x, w, temp, num_nodes, input_dim, output_dim);
    
    // Aggregation: AXW
    matmul(norm_adj, temp, out, num_nodes, num_nodes, output_dim);
    
    free(norm_adj);
    free(temp);
}